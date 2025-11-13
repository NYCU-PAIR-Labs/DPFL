from typing import List, Tuple
from .fedavg import FedAvg
import numpy as np
import copy
import torch
from tqdm import tqdm
from functools import reduce
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from colorama import Fore, Style
from utils.util import load_public_data

'''
Hyperparameters (From paper):
    - dataset: Digits (MNIST, USPS, SVHN, SYN)
    - model: ResNet10
    - optimizer: Adam
    - lr: 0.001
    - batch: 128, public_batch: 256
    - local_epoch: 20
    - public_epoch: 1
    - pretrain_epoch: 50
    - communication_rounds: 40 (MNIST) / 200 (FEMNIST)
    - num_clients: 4, (each for one domain)
    - off_diag_weight: 0.0051
'''

class FCCL(FedAvg):
    def __init__(self, device, args, params):
        super().__init__(device=device, args=args, params=params)
        self.prev_net = None       # local model of previous round
        self.intra_net = None       # local model pretrained on local dataset

        self.comm_epoch = args.num_rounds
        self.LOCAL_LR = args.lr
        self.PUBLIC_LR = args.lr
        self.local_lr = self.LOCAL_LR
        self.public_lr = self.PUBLIC_LR

        self.pretrain_epochs = 200
        self.public_batch_size = int(params["public_batch_size"])                # 256
        self.off_diag_weight = params["off_diag_weight"]                  # 0.0051

        self.kl_loss = torch.nn.KLDivLoss(reduction='batchmean').to(self.device)


    def _initialization(self, **kwargs) -> None:
        client_list, active_clients = kwargs["client_list"], kwargs["active_clients"]

        ''' 1. Pre-train on private dataset '''
        for i in range(len(client_list)):
            # client_list[i].set_parameters(self.get_parameters())
            client_list[i].strategy.prev_net = copy.deepcopy(client_list[i].model)
            private_loss, private_acc = client_list[i].strategy.pretrain(client_list[i].model, client_list[i].trainLoader, client_list[i].valLoader, client_list[i].optimizer)
            print(Fore.RED + "[Client {}, Pre-train on private set] loss: {:.4f}, accuracy: {:.4f}".format(i, private_loss, private_acc) + Fore.RESET)

        unlabeled_dataset = load_public_data(args=self.args)
        unlabeled_dataset = torch.utils.data.Subset(unlabeled_dataset, np.random.choice(len(unlabeled_dataset), 5000, replace=False))
        self.public_loader = DataLoader(unlabeled_dataset, batch_size=self.public_batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
        ''' 2. Col update '''
        self.col_update(communication_idx=0, active_clients=active_clients, client_list=client_list)


    def _server_train_func(self, cid, rounds, client_list, **kwargs) -> None:
        ''' Train function for the server to orchestrate the training process '''
        result = client_list[cid].train()
        train_loss, inter_loss, intra_loss, train_acc = result["train_loss"], result["inter_loss"], result["intra_loss"], result["train_acc"]
        print("[Client {}] loss: {:.4f}, inter_loss: {:.4f}, intra_loss: {:.4f}, acc: {:.4f}".format(cid, train_loss, inter_loss, intra_loss, train_acc))


    def _server_agg_func(self, rounds, client_list, active_clients, global_model):
        ''' Aggregate function for the server '''
        new_weights = self._aggregation(client_list)

        self.col_update(rounds, active_clients, client_list)
        return new_weights


    def _train(self, model, trainLoader, optimizer, num_epochs) -> Tuple[float, float]:
        ''' Train function for the client '''
        inter_net = self.prev_net
        intra_net = self.intra_net
        inter_net.eval()
        intra_net.eval()

        model.train()
        for i in range(num_epochs):
            total_loss, total_correct = [], []
            total_inter_loss, total_intra_loss = [], []
            for x, label in trainLoader:
                x, label = x.to(self.device), label.to(self.device)
                output, _ = model(x)

                predict = torch.argmax(output.data, 1)
                total_correct.append((predict == label).sum().item() / len(predict))

                loss_hard = F.cross_entropy(output, label)                          # ce_loss

                logsoft_outputs = F.log_softmax(output, dim=1)
                with torch.no_grad():
                    inter_outputs = F.softmax(inter_net(x)[0], dim=1)
                    intra_outputs = F.softmax(intra_net(x)[0], dim=1)

                inter_loss = self.kl_loss(logsoft_outputs, inter_outputs)
                intra_loss = self.kl_loss(logsoft_outputs, intra_outputs)
                dual_loss = inter_loss + intra_loss

                loss = loss_hard + dual_loss
                total_inter_loss.append(inter_loss.item())
                total_intra_loss.append(intra_loss.item())
                total_loss.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)               # clip the gradient to prevent exploding
                optimizer.step()
        
        self.prev_net.load_state_dict(model.state_dict())
        # self.prev_net = copy.deepcopy(model)
        # return np.mean(total_loss), np.mean(total_inter_loss), np.mean(total_intra_loss), np.mean(total_correct)
        return {
            "train_loss": np.mean(total_loss),
            "inter_loss": np.mean(total_inter_loss),
            "intra_loss": np.mean(total_intra_loss),
            "train_acc": np.mean(total_correct)
        }


    def col_update(self, communication_idx, active_clients, client_list):
        # self.public_lr = self.PUBLIC_LR * (1 - communication_idx / self.comm_epoch * 0.9)
        for batch_idx, (images, _) in enumerate(self.public_loader):
            '''
            Aggregate the output from participants
            '''
            linear_output_list = []
            linear_output_target_list = []
            images = images.to(self.device)

            for _, net_id in enumerate(active_clients):
                net = client_list[net_id].model
                net.train()
                linear_output, _ = net(images)
                linear_output_target_list.append(linear_output.clone().detach())
                linear_output_list.append(linear_output)

            '''
            Update Participants' Models via Col Loss
            '''
            for net_idx, net_id in enumerate(active_clients):
                net = client_list[net_id].model
                net.train()
                optimizer = torch.optim.SGD(net.parameters(), lr=self.public_lr, momentum=0.9, weight_decay=self.args.weight_decay)

                linear_output_target_avg_list = []
                for k in range(len(linear_output_target_list)):
                    linear_output_target_avg_list.append(linear_output_target_list[k])

                linear_output_target_avg = torch.mean(torch.stack(linear_output_target_avg_list), 0)
                linear_output = linear_output_list[net_idx]
                z_1_bn = (linear_output-linear_output.mean(0))/linear_output.std(0)
                z_2_bn = (linear_output_target_avg-linear_output_target_avg.mean(0))/linear_output_target_avg.std(0)
                c = z_1_bn.T @ z_2_bn
                c.div_(len(images))

                # if batch_idx == len(self.public_loader)-3:
                #     c_array = c.detach().cpu().numpy()
                #     self._draw_heatmap(c_array, self.NAME,communication_idx,net_idx)

                on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                off_diag = self._off_diagonal(c).add_(1).pow_(2).sum()
                optimizer.zero_grad()
                col_loss = on_diag + self.off_diag_weight * off_diag
                if batch_idx == len(self.public_loader)-1:
                    print('Round: '+str(communication_idx)+' Net: '+str(net_idx)+', Col: '+str(col_loss.item()))
                col_loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
                optimizer.step()
        return None


    def pretrain(self, model, loader, valLoader, optimizer):
        epochs = self.pretrain_epochs
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=2e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)

        best_model = copy.deepcopy(model.state_dict())
        best_acc, best_loss = 0, 0

        t = tqdm(range(epochs), desc="Pre-training", leave=False)
        for _ in t:
            total_correct, total_loss = [], []
            for x, label in loader:
                x, label = x.to(self.device), label.to(self.device)
                output, _ = model(x)

                predict = torch.argmax(output.data, 1)
                total_correct.append((predict == label).sum().item() / len(predict))

                loss = F.cross_entropy(output, label)
                total_loss.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()

            # test_loss, test_acc, _ = self._test(model, valLoader)
            test_result = self._test(model, valLoader)
            test_loss, test_acc = test_result["test_loss"], test_result["test_acc"]
            if test_acc > best_acc:
                best_acc = test_acc
                best_loss = test_loss
                best_model = copy.deepcopy(model.state_dict())
            
            lr_scheduler.step()
            t.set_postfix(loss="{:.4f}".format(np.mean(total_loss)), accuracy="{:.4f}".format(np.mean(total_correct)), test_loss="{:.4f}".format(test_loss), test_acc="{:.4f}".format(test_acc))
        model.load_state_dict(best_model)

        # initialize `intra_net` for each client
        self.intra_net = copy.deepcopy(model)
        return best_loss, best_acc
    

    def _off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


    def _test(self, model, testLoader) -> Tuple[float, float]:
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