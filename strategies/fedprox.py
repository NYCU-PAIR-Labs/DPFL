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
    - dataset: MNIST / FEMNIST
    - model: multinomial logistic regression (MNIST / FEMNIST)
    - optimizer: SGD
    - lr: 0.03 (MNIST) / 0.003 (FEMNIST)
    - batch: 10
    - local_epoch: 20
    - communication_rounds: 100 (MNIST) / 200 (FEMNIST)
    - num_clients: 10, total 1000 (uniformly sampling)`
    - aggregation: weighted sum proportional to the number of local data points
    - proximal_mu: tune from {0.001, 0.01, 0.1, 1.0}
'''

class FedProx(FedAvg):
    def __init__(self, device, args, params):
        super().__init__(device=device, args=args, params=params)
        self.proximal_mu = params["mu"]


    def _initialization(self, **kwargs) -> None:
        pass


    def _server_train_func(self, cid, rounds, client_list, **kwargs) -> None:
        ''' Train function for the server to orchestrate the training process '''
        result = client_list[cid].train()
        train_loss, train_acc = result["train_loss"], result["train_acc"]
        print("[Client {}] loss: {:.4f}, acc: {:.4f}".format(cid, train_loss, train_acc))


    def _server_agg_func(self, rounds, client_list, active_clients, global_model):
        ''' Aggregate function for the server '''
        new_weights = self._aggregation(client_list)
        return new_weights


    def _train(self, model, trainLoader, optimizer, num_epochs, **kwargs) -> Tuple[float, float]:
        ''' Train function for the client '''
        global_model = copy.deepcopy(model)

        model.train()
        for i in range(num_epochs):
            total_loss, total_correct = [], []
            for x, label in trainLoader:
                x, label = x.to(self.device), label.to(self.device)
                output, _ = model(x)

                predict = torch.argmax(output.data, 1)
                total_correct.append((predict == label).sum().item() / len(predict))

                loss = F.cross_entropy(output, label)

                proximal_term = 0.0
                for local_weights, global_weights in zip(model.parameters(), global_model.parameters()):
                    proximal_term += (local_weights - global_weights).norm(2)

                loss += (self.proximal_mu / 2) * proximal_term
                total_loss.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)               # clip the gradient to prevent exploding
                optimizer.step()
        
        return {
            "train_loss": np.mean(total_loss),
            "train_acc": np.mean(total_correct)
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