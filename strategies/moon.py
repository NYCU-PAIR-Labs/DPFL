from typing import List, Tuple, Dict, Any
from .fedavg import FedAvg
import numpy as np
import copy
import torch
from functools import reduce
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer

'''
Hyperparameters (From paper):
    - dataset: CIFAR-10 / CIFAR-100 / Tiny-ImageNet (Dirichlet distribution, a = 0.5)
    - model: 2-layer CNN (6, 16 channels) + 2-layer MLP (120, 84) | CIFAR-10
    - projection head: 2-layer MLP (output_dim: 256)
    - optimizer: SGD, momentum: 0.9
    - lr: 0.01
    - weight_decay: 0.00001
    - batch: 64
    - local_epoch: 10
    - communication_rounds: 100
    - num_clients: 10
    - temperature: 0.5
    - mu: 5
    - aggregation: weighted averaging
'''

class MOON(FedAvg):
    def __init__(self, device, args, params):
        super().__init__(device=device, args=args, params=params)
        self.prev_model = None
        self.cosine_sim = torch.nn.CosineSimilarity(dim=-1)
        self.T = params["T"]                       # temperature for contrastive loss
        self.mu = params["mu"]                     # weight of contrastive loss


    def _initialization(self, **kwargs) -> None:
        ''' Initialize previous model (prev_model) with local model '''
        client_list, global_model = kwargs["client_list"], kwargs["global_model"]

        for i in range(len(client_list)):
            client_list[i].strategy.prev_model = copy.deepcopy(global_model)


    def _server_train_func(self, cid, rounds, client_list, **kwargs) -> None:
        ''' Train function for the server to orchestrate the training process '''
        result = client_list[cid].train()
        train_loss, train_acc, ce_loss, con_loss = result["train_loss"], result["train_acc"], result["ce_loss"], result["con_loss"]
        print("[Client {}] loss: {:.4f}, ce_loss: {:.4f}, con_loss: {:.4f}, acc: {:.4f}".format(cid, train_loss, ce_loss, con_loss, train_acc))


    def _server_agg_func(self, rounds, client_list, active_clients, global_model):
        ''' Aggregate function for the server '''
        new_weights = self._aggregation(client_list)
        return new_weights


    def _train(self, model, trainLoader, optimizer, num_epochs, **kwargs) -> Tuple[float, float]:
        ''' Train function for the client '''
        global_model = copy.deepcopy(model).to(self.device)
        global_model.eval()
        self.prev_model.eval()

        model.train()
        for i in range(num_epochs):
            total_loss, total_correct = [], []
            ce_loss, con_loss = [], []
            for x, label in trainLoader:
                x, label = x.to(self.device), label.to(self.device)
                output, feature = model(x)

                predict = torch.argmax(output.data, 1)
                total_correct.append((predict == label).sum().item() / len(predict))

                # cross-entropy loss
                loss1 = F.cross_entropy(output, label)

                # get feature from global model and previous model
                with torch.no_grad():
                    _, feature_g = global_model(x)
                    _, feature_p = self.prev_model(x)

                positive_pair = self.cosine_sim(feature, feature_g).reshape(-1, 1)
                negative_pair = self.cosine_sim(feature, feature_p).reshape(-1, 1)

                logits = torch.cat((positive_pair, negative_pair), dim=1).to(self.device)
                logits /= self.T

                target = torch.zeros(x.size(0)).to(self.device).long()

                # contrastive loss
                loss2 = F.cross_entropy(logits, target)

                loss = loss1 + loss2 * self.mu
                total_loss.append(loss.item())
                ce_loss.append(loss1.item())
                con_loss.append(loss2.item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)               # clip the gradient to prevent exploding
                optimizer.step()

        # copy current model to prev_model
        self.prev_model.load_state_dict(model.state_dict())
        # return np.mean(total_loss), np.mean(total_correct), np.mean(ce_loss), np.mean(con_loss)
        return {
            "train_loss": np.mean(total_loss),
            "train_acc": np.mean(total_correct),
            "ce_loss": np.mean(ce_loss),
            "con_loss": np.mean(con_loss)
        }

    def _test(self, model, testLoader, payload=None) -> Tuple[float, float]:
        ''' Test function for the client '''
        total_loss, total_correct = [], []
        num_classes = self.args.num_classes
        correct_predictions = {i: 0 for i in range(num_classes)}
        total_counts = {i: 0 for i in range(num_classes)}

        model.eval()
        with torch.no_grad():
            for x, label in testLoader:
                x, label = x.to(self.device), label.to(self.device)
                output, _ = model(x)
                predict = torch.argmax(output.data, 1)
                total_correct.append((predict == label).sum().item() / len(predict))
                total_loss.append(F.cross_entropy(output, label).item())

                for i in range(num_classes):
                    mask = label == i
                    correct_predictions[i] += (predict[mask] == label[mask]).sum().item()
                    total_counts[i] += mask.sum().item()

        class_acc = {i: (correct_predictions[i] / total_counts[i]) if total_counts[i] > 0 else 0 for i in range(num_classes)}
        # return np.mean(total_loss), np.mean(total_correct), class_acc
        return {
            "test_loss": np.mean(total_loss),
            "test_acc": np.mean(total_correct),
            "class_acc": class_acc
        }
    

    def _get_local_payload(self) -> Dict[str, Any]:
        """
        Get the current payload from the strategy.
        """
        return self.local_payload