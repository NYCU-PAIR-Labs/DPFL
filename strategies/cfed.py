from typing import List, Tuple, Dict, Any
from .fedavg import FedAvg
import numpy as np
import copy
import torch
import torch.nn as nn
from functools import reduce
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from tqdm import tqdm
from utils.data_model import load_public_data
from utils.util import ConcatDataset, set_param

'''
Hyperparameters (From paper):
    - dataset: CIFAR10 / CIFAR100 / Caltech256
        1. training dataset and surrogate dataset are both divided into 200 shards,
           and each client is assigned 2 shards on each task and 2 shards as the local surrogate)
        2. server also selects 2 shards for server distillation
    - model: ResNet-18
    - optimizer: Adam
    - lr: 5e-5
    - batch: -
    - task_num: 2 / 3
    - local_epoch: 10 (Domain-IL) / 40 (Class-IL)
    - communication_rounds: 20
    - num_clients: 10, total 100 (uniformly sampling)
    - temperature: 2
'''

class CFeD(FedAvg):
    def __init__(self, device, args, params):
        super().__init__(device=device, args=args, params=params)
        self.distillation_loss = DistillationLoss()
        self.local_distill_epochs = int(params["local_d_epochs"])
        self.server_distill_epochs = int(params["server_d_epochs"])
        self.T = params["T"]
        self.alpha = params["alpha"]
        self.d_batch_size = int(params["d_batch_size"])

        self.unlabled_dataset = None
        self.reviewLoaders = None           # collection of reviewLoaders for server distillation
        self.reviewLoader = None            # individual reviewLoader for each client distillation
        self.server_optimizer = None
        self.local_weights = None


    def _initialization(self, **kwargs) -> None:
        client_list, global_model = kwargs["client_list"], kwargs["global_model"]

        ''' Prepare surrogate dataset for reviewing '''
        self.unlabeled_dataset = load_public_data(dataset=self.args.dataset, args=self.args)
        
        # random split the dataset into args.client_num + 1
        length = len(self.unlabeled_dataset) // (self.args.num_clients + 1)
        last_length = len(self.unlabeled_dataset) - (length * self.args.num_clients)

        datsets = torch.utils.data.random_split(self.unlabeled_dataset, [length] * (self.args.num_clients) + [last_length], generator=torch.Generator().manual_seed(42))
        self.reviewLoaders = {
            i: torch.utils.data.DataLoader(datsets[i], batch_size=self.d_batch_size, shuffle=True, num_workers=2)
            for i in range(self.args.num_clients)
        }
        # self.strategy.reviewLoaders[-1] = torch.utils.data.DataLoader(datsets[-1], batch_size=self.args.batch_size, shuffle=True, num_workers=2)
        self.dataset = datsets[-1]

        ''' Distribute the review dataset to each client '''
        for i in range(len(client_list)):
            client_list[i].strategy.reviewLoader = self.reviewLoaders[i]

        ''' Get client's validation data '''
        valset = [client_list[i].valLoader.dataset for i in range(len(client_list))]
        self.valLoader = {
            "total_val": DataLoader(ConcatDataset(valset), batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
        }

        self.server_optimizer = torch.optim.SGD(global_model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=self.args.weight_decay)
        self.local_models = []
        self.local_weights = []                            # store model weights for current task and review task (dtype: state_dict)


    def _server_train_func(self, cid, rounds, client_list, **kwargs) -> None:
        ''' Train function for the server to orchestrate the training process '''
        # train_loss, train_acc, dist_loss, dist_acc, model_for_current, model_for_review = self.client_list[i].train(rounds=rounds, prev_global_model=self.strategy.prev_global_model, task_id=self._current_tid)
        global_model = kwargs["global_model"]

        result = client_list[cid].train(rounds=rounds, prev_global_model=global_model, task_id=rounds)
        train_loss, train_acc, dist_loss, dist_acc, model_for_current, model_for_review = result["train_loss"], result["train_acc"], result["dist_loss"], result["dist_acc"], result["model_for_current"], result["model_for_review"]
        # self.local_models.append(model_for_current)
        # self.local_weights.append([model_for_current, len(client_list[cid].trainLoader.dataset)])

        # if rounds > 1:                                                # reviewing on old task (rounds > 1)
        #     self.local_models.append(model_for_review)
        #     self.local_weights.append([model_for_review, len(client_list[cid].trainLoader.dataset)])
        print("[Client {}] loss: {:.4f}, dist_loss: {:.4f}, acc: {:.4f}, dist_acc: {:.4f}".format(cid, train_loss, dist_loss, train_acc, dist_acc))


    def _server_agg_func(self, rounds, client_list, active_clients, global_model):
        ''' Aggregate function for the server '''
        new_weights = self._aggregation(client_list)

        ''' Prepare old global model for distillation '''
        if rounds > 1:
            self.local_models.append(copy.deepcopy(global_model))

        ''' Prepare current global model for distillation '''
        global_weights = self.average_weights(self.local_weights)
        set_param(global_model, global_weights)

        self.reviewLoader = DataLoader(
            GlobalDataSetSplit(self.dataset, self.local_models, self.device), batch_size=self.d_batch_size, shuffle=True)

        ''' Server Distillation '''
        # new_weights = self.server_distillation(global_model)
        self.local_weights = []                                    # clear local_weights
        self.local_models = []
        return new_weights


    def _train(self, model, trainLoader, optimizer, num_epochs, **kwargs) -> Tuple[float, float]:
        ''' Train function for the client '''
        prev_global_model, task_id = kwargs["prev_global_model"], kwargs["task_id"]

        prev_global_model.eval()

        ''' Prepare for distillation (review old task) '''
        model_for_review = copy.deepcopy(model)
        model_for_review.train()
        optimizer_for_review = torch.optim.SGD(model_for_review.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=self.args.weight_decay)

        ''' Train on current task '''
        model.train()
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)               # clip the gradient to prevent exploding
                optimizer.step()

        ''' Review on old tasks '''
        if task_id > 1:
            for j in range(self.local_distill_epochs):
                distill_loss, distill_correct = [], []
                for x, label in self.reviewLoader:
                    x, label = x.to(self.device), label.to(self.device)
                    output, _ = model_for_review(x)

                    predict = torch.argmax(output.data, 1)
                    distill_correct.append((predict == label).sum().item() / len(predict))

                    ''' Distillation from previous model (`t-1` global model) '''
                    with torch.no_grad():
                        old_output, _ = prev_global_model(x)
                    loss = self.distillation_loss(output, old_output, temperature=self.T, frac=self.alpha)
                    distill_loss.append(loss.item())

                    optimizer_for_review.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)               # clip the gradient to prevent exploding
                    optimizer_for_review.step()
            model_for_review.cpu()
        else:
            distill_loss, distill_correct = [0], [0]

        # return np.mean(total_loss), np.mean(total_correct), np.mean(distill_loss), np.mean(distill_correct), model, model_for_review
        return {
            "train_loss": np.mean(total_loss),
            "train_acc": np.mean(total_correct),
            "dist_loss": np.mean(distill_loss),
            "dist_acc": np.mean(distill_correct),
            "model_for_current": model,
            "model_for_review": model_for_review
        }


    def server_distillation(self, global_model):
        ''' Train function for the client '''
        # model_list = [client_list[idx].model for idx in active_client_list] + [old_global_model]
        # id_list = [client_list[idx].cid for idx in active_client_list] + [-1]

        intial_val_acc = self.evaluate(global_model, self.valLoader)      # inital val_acc
        state = {
            "best_val_acc": 0,
            "best_server_model": copy.deepcopy(global_model.state_dict()),
        }

        print("\n[Server | Distillation] Start distillation, inital val_acc: {:.4f}".format(intial_val_acc))
        pbar = tqdm(range(self.server_distill_epochs), desc="[Server | Distillation]", leave=False)
        
        for n in pbar:
            global_model.train()
            total_loss = []

            ''' Distillation using each client's reviewLoader and its model '''
            for x, soft_labels in self.reviewLoader:
                x, soft_labels = x.to(self.device), soft_labels.to(self.device)
                output, _ = global_model(x)

                loss = self.distillation_loss(output, soft_labels, temperature=self.T, frac=self.alpha)
                total_loss.append(loss.item())

                self.server_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(global_model.parameters(), 10)
                self.server_optimizer.step()

            val_acc = self.evaluate(global_model, self.valLoader)
            pbar.set_postfix({"val_acc": val_acc})
            if val_acc > state["best_val_acc"]:
                state["best_val_acc"] = val_acc
                state["best_server_model"] = copy.deepcopy(global_model.state_dict())

        # restore the best model
        global_model.load_state_dict(state["best_server_model"])
        print("[Server | Distillation] Best val_acc: {:.4f}".format(state["best_val_acc"]))
        return global_model.state_dict().values()


    def average_weights(self, local_weights: List[dict]) -> dict:
        ''' Average weights
            Args:
                w: list of weights (state_dict)
            Returns:
                w_avg: averaged weights (state_dict)
        '''

        client_numData_model_pair = [
            (model, numData) for model, numData in local_weights
        ]
        num_examples_total = sum([num_examples for _, num_examples in client_numData_model_pair])
        weighted_weights = [
            [model.state_dict()[params] * num_examples for params in model.state_dict()] for model, num_examples in client_numData_model_pair
        ]

        # Compute average weight of each layer
        weights_prime = [
            reduce(torch.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
        return weights_prime


    def evaluate(self, server_model, testLoader) -> Tuple[float, float]:
        avg_acc = []
        for name, loader in testLoader.items():
            # _, acc_indi, _ = self._test(server_model, loader)
            acc_indi = self._test(server_model, loader)["test_acc"]
            avg_acc.append(acc_indi)
        return np.mean(avg_acc)


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


    # def get_reviewLoader(self):
    #     ''' Sample 5000 data from public dataset for knowledge distillation '''
    #     public_subset = torch.utils.data.Subset(self.unlabled_dataset, np.random.choice(len(self.unlabled_dataset), 2000, replace=False))
    #     loader = torch.utils.data.DataLoader(public_subset, batch_size=self.args.batch_size, shuffle=True, num_workers=2)
    #     return loader


class DistillationLoss(nn.Module):
    def __init__(self):
        super(DistillationLoss, self).__init__()

    def forward(self, output, old_target, temperature, frac):
        T = temperature
        alpha = frac
        outputs_S = F.log_softmax(output / T, dim=1)
        outputs_T = F.softmax(old_target / T, dim=1)
        l_old = outputs_T.mul(outputs_S)
        l_old = -1.0 * torch.sum(l_old) / outputs_S.shape[0]

        return l_old * alpha
    

class GlobalDataSetSplit(torch.utils.data.Dataset):
    def __init__(self, dataset, local_models, device):
        self.device = device
        self.dataset = dataset
        # self.idxs = [int(i) for i in idxs]
        self.local_models = local_models
        for i in self.local_models:
            i.eval()
        self.model_idx = 0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        temp_image, label = image.to(self.device), torch.tensor(label).to(self.device)
        temp_image = temp_image.unsqueeze(0)
        soft_label, _ = self.local_models[self.model_idx](temp_image)
        self.model_idx = (self.model_idx + 1) % len(self.local_models)
        return image, soft_label.view(soft_label.shape[1])