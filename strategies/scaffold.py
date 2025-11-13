from typing import List, Tuple, Dict, Any
from numpy import ndarray
from .strategy import Strategy
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
    - dataset: EMNIST
    - model: logistic regression / 2-layer MLP
    - optimizer: SGD
    - lr: -
    - batch: 0.2 of local data
    - local_epoch: {1, 5, 10, 20}
    - local step: 5
    - communication_rounds: -
    - num_clients: total 100, sampling rate 20%
    - aggregation: averaged sum
'''

'''
Hyperparameters (From NIID-Bench):
    - dataset: CIFAR10
    - model: 2-layer CNN + 2-layer MLP
    - optimizer: SGD
    - lr: 0.01, momentum: 0.9
    - batch: 64
    - local_epoch: 10
    - communication_rounds: 50
    - num_clients: 10
'''

class SCAFFOLD(FedAvg):
    def __init__(self, device, args, params):
        super().__init__(device=device, args=args, params=params)
        self.global_payload = None
        self.c_local = None
        self.c_delta = None
        self.local_payload["type"] = "variate"

        # self.n_minibatch = 50       # from FedFTG's implementation


    def _initialization(self, **kwargs) -> None:
        ''' Initialize the c_global & c_local state '''
        client_list, global_model = kwargs["client_list"], kwargs["global_model"]

        self.global_payload = copy.deepcopy(global_model)                       # construct c_global
        for i in range(len(client_list)):
            client_list[i].strategy.c_local = copy.deepcopy(global_model)    # construct c_local for each client


    def _server_train_func(self, cid, rounds, client_list, **kwargs) -> None:
        ''' Train function for the server to orchestrate the training process '''
        result = client_list[cid].train(c_global=self.global_payload)
        train_loss, train_acc = result["train_loss"], result["train_acc"]
        print("[Client {}] loss: {:.4f}, acc: {:.4f}".format(cid, train_loss, train_acc))
    

    def _server_agg_func(self, rounds, client_list, active_clients, global_model):
        ''' Aggregate function for the server '''
        new_weights = self.aggregate(
            c_global=self.global_payload,
            client_list=client_list,
            active_list=active_clients,
        )
        return new_weights


    def _train(self, model, trainLoader, optimizer, num_epochs, **kwargs) -> Tuple[float, float]:
        ''' Train function for the client '''
        c_global = kwargs["c_global"]

        global_model = copy.deepcopy(model).to(self.device)
        c_local = self.c_local.to(self.device)
        c_global = c_global.to(self.device)

        c_local_para = c_local.state_dict()
        c_global_para = c_global.state_dict()

        model.train()
        k = 0

        # total_loss, total_correct = [], []
        # while k < self.n_minibatch:
        #     try:
        #         x, label = next(self.data_iter)
        #     except:
        #         self.data_iter = iter(trainLoader)
        #         x, label = next(self.data_iter)

        for i in range(num_epochs):
            total_loss, total_correct = [], []
            for x, label in trainLoader: 
                x, label = x.to(self.device), label.to(self.device)
                output, _ = model(x)

                predict = torch.argmax(output.data, 1)
                total_correct.append((predict == label).sum().item() / len(predict))

                loss = F.cross_entropy(output, label)
                total_loss.append(loss.item())

                optimizer.zero_grad()
                loss.backward()

                # update local model
                for key in model.state_dict():
                    if model.state_dict()[key].grad is not None:
                        model.state_dict()[key].grad += (c_global_para[key] - c_local_para[key]).data

                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)               # clip the gradient to prevent exploding
                optimizer.step()
                k += 1

        # update c
        c_new_para = c_local.state_dict()
        c_delta_para = copy.deepcopy(c_local.state_dict())

        global_para = global_model.state_dict()
        local_para = model.state_dict()

        for key in local_para:
            c_new_para[key] = c_new_para[key] - c_global_para[key] + (global_para[key] - local_para[key]) / (k * self.args.lr)
            c_delta_para[key] = c_new_para[key] - c_local_para[key]

        self.c_local.load_state_dict(c_new_para)
        self.c_delta = c_delta_para
        self.local_payload["data"] = self.c_delta

        # return np.mean(total_loss), np.mean(total_correct)
        return {
            "train_loss": np.mean(total_loss),
            "train_acc": np.mean(total_correct)
        }


    def aggregate(self, c_global, client_list, active_list):
        # c_global_para = c_global.state_dict()

        # # calculate c_delta (c_new - c_old)
        # for i in active_list:
        #     for key in c_global_para:
        #         if c_global_para[key].type() == 'torch.LongTensor':
        #             c_global_para[key] += (client_list[i].strategy.c_delta[key] / self.args.num_clients).type(torch.LongTensor)
        #         elif c_global_para[key].type() == 'torch.cuda.LongTensor':
        #             c_global_para[key] += (client_list[i].strategy.c_delta[key] / self.args.num_clients).type(torch.cuda.LongTensor)
        #         else:
        #             c_global_para[key] += client_list[i].strategy.c_delta[key] / self.args.num_clients

        # # update c_global
        # self.global_payload.load_state_dict(c_global_para)
        new_weights = super()._aggregation(client_list, average_mode="uniform")
        return new_weights
    

    def payload_aggregation(self, local_payload_list, client_weight_list):
        c_global_para = self.global_payload.state_dict()
        for i in range(len(local_payload_list)):
            for key in local_payload_list[i]:
                if c_global_para[key].type() == 'torch.LongTensor':
                    c_global_para[key] += (local_payload_list[i][key] * client_weight_list[i]).type(torch.LongTensor)
                elif c_global_para[key].type() == 'torch.cuda.LongTensor':
                    c_global_para[key] += (local_payload_list[i][key] * client_weight_list[i]).type(torch.cuda.LongTensor)
                else:
                    c_global_para[key] += (local_payload_list[i][key] * client_weight_list[i])

        self.global_payload.load_state_dict(c_global_para)
        return self.global_payload


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