from typing import List, Tuple, Dict, Any
from .fedavg import FedAvg
from colorama import Fore
import numpy as np
import copy
import torch
from tqdm import tqdm
from functools import reduce
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from utils.util import ConcatDataset, set_param

'''
Hyperparameters (From paper):
    - dataset: MNIST / FEMNIST
    - model: multinomial logistic regression (MNIST / FEMNIST)
    - optimizer: SGD
    - lr: 0.03 (MNIST) / 0.003 (FEMNIST)
    - batch: 10
    - local_epoch: 20
    - communication_rounds: 100 (MNIST) / 200 (FEMNIST)
    - num_clients: 10, total 1000 (uniformly sampling)
    - aggregation: weighted sum proportional to the number of local data points
    - proximal_mu: tune from {0.001, 0.01, 0.1, 1.0}
'''

class FedCL(FedAvg):
    def __init__(self, device, args, params):
        super().__init__(device=device, args=args, params=params)
        self.w_d = None
        self.proxy_ratio = params["proxy_ratio"]
        self.coe = params["coe"]


    def _initialization(self, **kwargs) -> None:
        client_list, global_model = kwargs["client_list"], kwargs["global_model"]

        self.optimizer_server = torch.optim.SGD(global_model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.w_d = None

        ''' Compute client's total training data samples '''
        total_trainset_num = sum([len(client_list[i].trainLoader.dataset) for i in range(len(client_list))])
        proxy_size = int(total_trainset_num * self.proxy_ratio)

        ''' Initialize the proxy dataset used for EWC '''
        tmp_dataset = ConcatDataset([client_list[i].valLoader.dataset for i in range(len(client_list))])
        proxy_dataset = torch.utils.data.Subset(tmp_dataset, np.random.choice(len(tmp_dataset), proxy_size, replace=False))
        self.proxyLoader = DataLoader(proxy_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        print(Fore.RED + "[Server] Total trainset: {}, proxy_size: {}".format(total_trainset_num, proxy_size) + Fore.RESET)


    def _server_train_func(self, cid, rounds, client_list, **kwargs) -> None:
        ''' Train function for the server to orchestrate the training process '''
        result = client_list[cid].train(w_d=self.w_d)
        train_loss, ewc_loss, train_acc = result["train_loss"], result["ewc_loss"], result["train_acc"]
        print("[Client {}] loss: {:.4f}, ewc_loss: {:.4f}, acc: {:.4f}".format(cid, train_loss, ewc_loss, train_acc))


    def _server_agg_func(self, rounds, client_list, active_clients, global_model):
        ''' Aggregate function for the server '''
        new_weights = self._aggregation(client_list)

        set_param(global_model, new_weights)                                    # set global model's weight for EWC
        self.w_d = self.ewc(global_model, self.proxyLoader, self.optimizer_server)
        return new_weights


    def _train(self, model, trainLoader, optimizer, num_epochs, **kwargs) -> Tuple[float, float]:
        ''' Train function for the client '''
        w_d = kwargs["w_d"]

        global_model = copy.deepcopy(model).state_dict()

        model.train()
        for i in range(num_epochs):
            total_loss, total_correct = [], []
            ewc_loss = []
            for x, label in trainLoader:
                x, label = x.to(self.device), label.to(self.device)
                output, _ = model(x)

                predict = torch.argmax(output.data, 1)
                total_correct.append((predict == label).sum().item() / len(predict))

                ce_loss = F.cross_entropy(output, label)

                cons_loss = 0
                if w_d is not None:
                    for k, v in model.named_parameters():
                        cons_loss += w_d[k] * torch.sum((global_model[k] - v) ** 2)
                    cons_loss = (cons_loss * self.coe).item()

                loss = ce_loss + cons_loss
                ewc_loss.append(cons_loss)
                total_loss.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)               # clip the gradient to prevent exploding
                optimizer.step()
        # return np.mean(total_loss), np.mean(ewc_loss), np.mean(total_correct)
        return {
            "train_loss": np.mean(total_loss),
            "ewc_loss": np.mean(ewc_loss),
            "train_acc": np.mean(total_correct)
        }


    def ewc(self, model, data_loader, optimizer):
        tmp_weights = dict()
        for k, p in model.named_parameters():
            tmp_weights[k] = torch.zeros_like(p)

        model.eval()
        num_examples = 0
        for x, label in tqdm(data_loader, desc="[Server | EWC]", leave=False):
            x, label = x.to(self.device), label.to(self.device)
            num_examples += x.size(0)

            # compute output
            output, _ = model(x)
            loss = F.cross_entropy(output, label)

            optimizer.zero_grad()
            loss.backward()

            for k, p in model.named_parameters():
                tmp_weights[k].add_(p.grad.detach() ** 2)
        
        for k, v in tmp_weights.items():
            tmp_weights[k] = torch.sum(v).div(num_examples)

        return tmp_weights


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