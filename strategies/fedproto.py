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
    - dataset: MNIST / FEMNIST / CIFAR-10 (n-way k-shot)
    - model: ResNet18 | CIFAR-10
    - optimizer: SGD, momentum: 0.5
    - lr: 0.01
    - weight_decay: SGD: 0 / Adam: 1e-4
    - batch: -
    - local_epoch: 1
    - ld: 1
    - communication_rounds: 100
    - num_clients: 20
'''

class FedProto(FedAvg):
    def __init__(self, device, args, params):
        super().__init__(device=device, args=args, params=params)
        self.local_protos = {}                                          # original local prototypes of FedProto
        self.local_payload["type"] = "protos"                           # Type of payload (e.g., logits, prototypes)
        self.ld = params["ld"]               # weight for prototype loss


    def _initialization(self, **kwargs) -> None:
        pass


    def _server_train_func(self, cid, rounds, client_list, **kwargs) -> None:
        ''' Train function for the server to orchestrate the training process '''
        result = client_list[cid].train(global_protos=kwargs["global_payload"])
        train_loss, ce_loss, proto_loss, train_acc, train_acc_proto = result["train_loss"], result["ce_loss"], result["proto_loss"], result["train_acc"], result["proto_acc"]
        self.local_protos[client_list[cid].cid] = client_list[cid].strategy.local_payload["data"]           # save local prototypes
        print("[Client {}] loss: {:.4f}, ce_loss: {:.4f}, proto_loss: {:.4f}, acc: {:.4f}, proto_acc: {:.4f}".format(cid, train_loss, ce_loss, proto_loss, train_acc, train_acc_proto))


    def _server_agg_func(self, rounds, client_list, active_clients, global_model):
        ''' Aggregate function for the server '''
        new_weights = self._aggregation(client_list)

        self.global_payload = self.aggregate_protos(self.local_protos)
        self.local_protos = {}                                     # clear local_protos
        return new_weights


    def _train(self, model, trainLoader, optimizer, num_epochs, **kwargs) -> Tuple[float, float]:
        ''' Train function for the client 
        local_protos {} (存每一個 user 的 local prototypes): {
            user_idx: [protos],
        }

        protos: {
            `label`: proto_list,
            0: [proto1, proto2, proto3],
            1: [proto1, proto2, proto3],
        }
        '''
        global_protos = kwargs["global_protos"]

        model.train()
        for i in range(num_epochs):
            total_loss, total_correct_ce, total_correct_proto = [], [], []
            ce_loss, proto_loss = [], []
            for x, labels in trainLoader:
                x, labels = x.to(self.device), labels.to(self.device)
                output, protos = model(x)

                predict = torch.argmax(output.data, 1)
                total_correct_ce.append((predict == labels).sum().item() / len(predict))

                # loss1: cross-entropy loss
                loss1 = F.cross_entropy(output, labels)
                
                # loss2: prototype distance loss
                if len(global_protos) == 0:
                    loss2 = 0 * loss1
                    total_correct_proto = 0
                else:
                    # compute the dist between protos and global_protos
                    a_large_num = 100
                    dist = a_large_num * torch.ones(size=(x.shape[0], self.args.num_classes)).to(self.device)  # initialize a distance matrix
                    for k in range(x.shape[0]):
                        for j in range(self.args.num_classes):
                            if j in global_protos.keys():
                                d = F.mse_loss(protos[k, :], global_protos[j][0])
                                dist[k, j] = d

                    # prediction
                    predict = torch.argmin(dist, 1)
                    total_correct_proto.append((predict == labels).sum().item() / len(predict))
                    
                    proto_new = copy.deepcopy(protos.data)
                    for idx, label in enumerate(labels):
                        if label.item() in global_protos.keys():
                            proto_new[idx, :] = global_protos[label.item()][0].data
                    loss2 = F.mse_loss(proto_new, protos)

                loss = loss1 + loss2 * self.ld
                ce_loss.append(loss1.item())
                proto_loss.append(loss2.item())
                total_loss.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)               # clip the gradient to prevent exploding
                optimizer.step()

        # generate local prototypes
        model.eval()
        agg_protos_label = {}
        with torch.no_grad():
            for x, labels in trainLoader:
                x, labels = x.to(self.device), labels.to(self.device)
                output, protos = model(x)
                for k in range(len(labels)):
                    if labels[k].item() in agg_protos_label:
                        agg_protos_label[labels[k].item()].append(protos[k,:])
                    else:
                        agg_protos_label[labels[k].item()] = [protos[k,:]]

        self.local_payload["data"] = self.agg_func(agg_protos_label)
        # return np.mean(total_loss), np.mean(ce_loss), np.mean(proto_loss), np.mean(total_correct_ce), np.mean(total_correct_proto)
        return {
            "train_loss": np.mean(total_loss),
            "ce_loss": np.mean(ce_loss),
            "proto_loss": np.mean(proto_loss),
            "train_acc": np.mean(total_correct_ce),
            "proto_acc": np.mean(total_correct_proto)
        }


    def _test(self, model, testLoader, payload=None) -> Tuple[float, float]:
        ''' Test function for the client '''
        global_protos = payload

        total_loss, total_correct_ce, total_correct_proto = [], [], []
        num_classes = self.args.num_classes
        correct_predictions = {i: 0 for i in range(num_classes)}
        total_counts = {i: 0 for i in range(num_classes)}

        model.eval()
        with torch.no_grad():
            if len(global_protos) != 0:
                for batch_idx, (images, labels) in enumerate(testLoader):
                    images, labels = images.to(self.device), labels.to(self.device)
                    output, protos = model(images)

                    # prediction by cross-entropy loss
                    # predict_ce = torch.argmax(output.data, 1)
                    # total_correct_ce.append((predict_ce == labels).sum().item() / len(predict_ce))

                    # compute the dist between protos and global_protos
                    a_large_num = 100
                    dist = a_large_num * torch.ones(size=(images.shape[0], self.args.num_classes)).to(self.device)  # initialize a distance matrix
                    for i in range(images.shape[0]):
                        for j in range(self.args.num_classes):
                            if j in global_protos.keys():
                                d = F.mse_loss(protos[i, :], global_protos[j][0])
                                dist[i, j] = d

                    # prediction by prototype distance
                    predict = torch.argmin(dist, 1)
                    total_correct_proto.append((predict == labels).sum().item() / len(predict))

                    for i in range(num_classes):
                        mask = labels == i
                        correct_predictions[i] += (predict[mask] == labels[mask]).sum().item()
                        total_counts[i] += mask.sum().item()

                    # compute loss
                    proto_new = copy.deepcopy(protos.data)
                    for idx, label in enumerate(labels):
                        if label.item() in global_protos.keys():
                            proto_new[idx, :] = global_protos[label.item()][0].data
                    total_loss.append(F.mse_loss(proto_new, protos).item())

            class_acc = {i: (correct_predictions[i] / total_counts[i]) if total_counts[i] > 0 else 0 for i in range(num_classes)}
        # return np.mean(total_loss), np.mean(total_correct_proto), class_acc
        return {
            "test_loss": np.mean(total_loss),
            "test_acc": np.mean(total_correct_proto),
            "class_acc": class_acc
        }


    def aggregate_protos(self, client_protos):
        """ calculate global prototypes """
        global_protos = self.proto_aggregation(client_protos)
        return global_protos
    

    def agg_func(self, local_protos):
        """ Aggregate each local prototypes.
        local_protos: {
            `label`: proto_list,
            0: [proto1, proto2, proto3],
            1: [proto1, proto2, proto3],
        }
        Returns the average of the weights.
        """
        for [label, proto_list] in local_protos.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                local_protos[label] = proto / len(proto_list)
            else:
                local_protos[label] = proto_list[0]
        return local_protos
    

    def proto_aggregation(self, local_protos_list):
        """ Aggregate all local prototypes """
        agg_protos_label = dict()
        for idx in local_protos_list:
            local_protos = local_protos_list[idx]
            for label in local_protos.keys():
                if label in agg_protos_label:
                    agg_protos_label[label].append(local_protos[label])
                else:
                    agg_protos_label[label] = [local_protos[label]]

        for [label, proto_list] in agg_protos_label.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0].data
                for i in proto_list:
                    proto += i.data
                agg_protos_label[label] = [proto / len(proto_list)]
            else:
                agg_protos_label[label] = [proto_list[0].data]
        return agg_protos_label
    

    def payload_aggregation(self, local_payload_list, client_weight_list):
        """ Aggregate all local prototypes by client class-wise weights. """
        # print(client_weight_list)

        agg_protos_label = dict()
        for idx in local_payload_list:
            local_protos = local_payload_list[idx]
            w = client_weight_list[idx]                                                 # Use client weight to aggregate prototypes
            for label in local_protos.keys():
                if label in agg_protos_label:
                    agg_protos_label[label].append((local_protos[label], w))
                else:
                    agg_protos_label[label] = [(local_protos[label], w)]

        for [label, proto_list] in agg_protos_label.items():
            if len(proto_list) > 1:
                proto = 0 * proto_list[0][0].data
                for i in proto_list:
                    proto += i[0].data * i[1]  # weighted sum
                agg_protos_label[label] = [proto / sum([i[1] for i in proto_list])]
            else:
                agg_protos_label[label] = [proto_list[0][0].data]
        return agg_protos_label
    

    def _get_local_payload(self) -> Dict[str, Any]:
        """
        Get the current payload from the strategy.
        """
        return self.local_payload