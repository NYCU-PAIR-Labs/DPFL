from typing import List, Tuple
from .fedavg import FedAvg
from .scaffold import SCAFFOLD
import numpy as np
import copy
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.model import CGenerator
from utils.util import set_param

'''
Hyperparameters (From paper):
    - dataset: CIFAR10 / CIFAR100, (Dirichlet distribution, a=0.3, 0.6)
    - model: ResNet18
    - batch: 50
    - lr_decay: 0.998
    - weight_decay: 0.001
    - local_epoch: 5
    - communication_rounds: 1000
    - num_clients: 10, total 100 (sample rate: 0.1)

    Local training (client):
        - lr: 0.1
    Classifier (global model):
        - optimizer: SGD
        - lr: 0.1
    Generator:
        - optimizer: Adam
        - lr: 0.01

    - z_dim: 100 (CIFAR10) / 256 (CIFAR100)
'''

class FedFTG(SCAFFOLD):
    def __init__(self, device, args, params):
        super().__init__(device=device, args=args, params=params)
        ''' Parameters for local training '''
        ''' Parameters for data-free knowledge distillation '''
        self.cgan = None
        self.optimizer_cgan = None
        self.scheduler_cgan = None
        ''' Hyperparameters for knowledge distillation'''
        self.iterations = 10
        self.inner_round_g = 1
        self.inner_round_d = 5
        self.z_dim = 100
        ''' Parameters for client data statistics '''
        self.client_class_num = []
    

    def _initialization(self, **kwargs) -> None:
        ''' Initialize the c_global & c_local state '''
        client_list, global_model = kwargs["client_list"], kwargs["global_model"]

        self.n_clnt = self.args.num_clients
        self.n_par = len(self.get_mdl_params([global_model])[0])
        self.idx_nonbn = self.get_mdl_nonbn_idx([global_model])[0]
        self.state_params_diffs = np.zeros((self.n_clnt + 1, self.n_par)).astype('float32')     # including cloud state

        weight_list = np.asarray([len(client_list[i].trainLoader.dataset) for i in range(len(client_list))])
        self.weight_list = weight_list / np.sum(weight_list) * self.n_clnt                               # normalize it
        self.delta_c_sum = np.zeros(self.n_par)

        ''' Initialize CGAN for FedFTG '''
        self.cgan = CGenerator().to(self.device)
        self.optimizer_cgan = torch.optim.Adam(self.cgan.parameters(), lr=0.01)
        self.scheduler_cgan = torch.optim.lr_scheduler.StepLR(self.optimizer_cgan, step_size=1, gamma=0.998)
        
        ''' Create lr_scheduler for global model '''
        self.optimizer_server = torch.optim.SGD(global_model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.scheduler_server = torch.optim.lr_scheduler.StepLR(self.optimizer_server, step_size=1, gamma=0.998)

        ''' Compute the class distributions of each client for Customized Label Sampling '''
        client_class_num = np.zeros((len(client_list), self.args.num_classes))
        for i in range(len(client_list)):
            for _, label in client_list[i].trainLoader.dataset:
                client_class_num[i, label] += 1
        self.client_class_num = client_class_num
        self.testLoader = copy.deepcopy(self.testLoader)


    def _server_train_func(self, cid, rounds, client_list, **kwargs) -> None:
        ''' Train function for the server to orchestrate the training process '''
        # Scale down c
        state_params_curr = torch.tensor(-self.state_params_diffs[cid] + self.state_params_diffs[-1] / self.weight_list[cid], dtype=torch.float32, device=self.device)
        result = client_list[cid].train(state_params_diff_curr=state_params_curr[self.idx_nonbn], state_params_diffs=self.state_params_diffs)
        train_loss, train_acc, new_c = result["train_loss"], result["train_acc"], result["new_c"]

        # Scale up delta c
        self.delta_c_sum += (new_c - self.state_params_diffs[cid]) * self.weight_list[cid]
        self.state_params_diffs[cid] = new_c
        print("[Client {}] loss: {:.4f}, acc: {:.4f}".format(cid, train_loss, train_acc))


    def _server_agg_func(self, rounds, client_list, active_clients, global_model):
        ''' Aggregate function for the server '''
        new_weights = self._aggregation(client_list)

        self.state_params_diffs[-1] += 1 / self.n_clnt * self.delta_c_sum
        self.delta_c_sum = np.zeros(self.n_par)           # clear delta_c_sum

        set_param(global_model, new_weights)                                    # set global model's weight for distillation
        
        ''' Test performance on selected models '''
        self.evaluate(rounds=rounds, model=global_model, tag="selected")
                                        
        loss_g, loss_md, loss_cls, loss_div, loss_d, avg_acc, new_weights = self.data_free_knowledge_distillation(global_model, client_list, active_clients, self.optimizer_server, self.scheduler_server)
        print("[Server] loss_D: {:.4f}, loss_G: {:.4f}, loss_md: {:.4f}, loss_cls: {:.4f}, loss_div: {:.4f}".format(loss_d, loss_g, loss_md, loss_cls, loss_div))
        # wandb.log({"avg_acc": avg_acc, "rounds": rounds})
        return new_weights


    def data_free_knowledge_distillation(self, global_model, client_list, active_client_list, optimizer_server, scheduler_server):
        server = global_model
        generator = self.cgan
        # optimizer_server = torch.optim.SGD(server.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        bs = self.args.batch_size

        ''' Coefficients for individual loss '''
        lambda_cls = 1.0
        lambda_div = 1.0

        ''' Sample random classes according to label distributions '''
        # print("Active client list:", active_client_list)
        clnt_cls_num = self.client_class_num[active_client_list]
        num_clients, num_classes = clnt_cls_num.shape
        cls_num = np.sum(clnt_cls_num, axis=0)
        # cls_clnt_weight = torch.from_numpy(clnt_cls_num / (np.tile(cls_num[np.newaxis, :], (num_clients, 1)) + 1e-6)).type(torch.float32).T
        cls_clnt_weight = clnt_cls_num / (np.tile(cls_num[np.newaxis, :], (num_clients, 1)) + 1e-6)
        cls_clnt_weight = cls_clnt_weight.transpose()
        labels_all = self.generate_labels(self.iterations * bs, cls_num)

        # labels_batches = torch.split(labels_all, bs)

        avg_acc = self.evaluate(server, self.testLoader)      # inital test_acc (avg_acc)
        print("\n[Server | Knowledge Distilaltion] Inital test_acc: {:.4f}".format(avg_acc))
        
        pbar = tqdm(range(self.iterations), desc="[Server | Knowledge Distillation]", leave=False)
        for i in pbar:
            # labels = labels_batches[i]                        # labels = labels_all[e*batch_size:(e*batch_size+batch_size)]
            labels = labels_all[i*bs:(i*bs+bs)]
            batch_weight = torch.tensor(self.get_batch_weight(labels, cls_clnt_weight)).to(self.device)
            # y_onehot = F.one_hot(labels, num_classes=num_classes).type(torch.float32).to(self.device)
            onehot = np.zeros((bs, num_classes))
            onehot[np.arange(bs), labels] = 1
            y_onehot = torch.Tensor(onehot).cuda()
            z = torch.randn((bs, self.z_dim, 1, 1), device=self.device)

            ''' Train Generator '''
            generator.train()
            server.eval()
            loss_G_total = 0
            loss_md_total = 0
            loss_cls_total = 0
            loss_div_total = 0

            for _ in range(self.inner_round_g):
                for k, idx in enumerate(active_client_list):
                    self.optimizer_cgan.zero_grad()

                    _teacher = client_list[idx].model
                    _teacher.eval()
                    loss, loss_md, loss_cls, loss_div = self.train_generator(
                            z=z,
                            y_onehot=y_onehot,
                            labels=labels,
                            generator=generator,
                            student=server,
                            teacher=_teacher,
                            weight=batch_weight[:, k],
                            num_clients=num_clients,
                        )

                    # loss_md_total.append(loss_md.item())
                    # loss_cls_total.append(loss_cls.item())
                    # loss_div_total.append(loss_div.item())
                    loss_md_total += loss_md.item()
                    loss_cls_total += loss_cls.item()
                    loss_div_total += loss_div.item()

                    # print("loss_md: {:.4f}, loss_cls: {:.4f}, loss_div: {:.4f}".format(loss_md.item(), loss_cls.item(), loss_div.item()))
                    # print(batch_weight[:, k])

                    # loss = loss_md + lambda_cls * loss_cls  + lambda_div * loss_div
                    # loss_G_total.append(loss.item())
                    loss_G_total += loss.item()

                    # self.optimizer_cgan.zero_grad()
                    # loss.backward()
                    self.optimizer_cgan.step()

            ''' Train student (global model) '''
            generator.eval()
            server.train()
            loss_D_total = 0

            for _ in range(self.inner_round_d):
                optimizer_server.zero_grad()
                fake = generator(z, y_onehot).detach()
                student_logit, _ = server(fake)
                t_logit_merge = 0
                for n , idx in enumerate(active_client_list):
                    _teacher = client_list[idx].model
                    _teacher.eval()
                    teacher_logit, _ = _teacher(fake)
                    teacher_logit = teacher_logit.detach()
                    t_logit_merge += F.softmax(teacher_logit, dim=1) * batch_weight[:, n][:, np.newaxis].repeat(1, self.args.num_classes)
                loss_D = torch.mean(-F.log_softmax(student_logit, dim=1) * t_logit_merge)
                # loss_D_total.append(loss_D.item())
                loss_D_total += loss_D.item()

                # optimizer_server.zero_grad()
                loss_D.backward()
                optimizer_server.step()

            pbar.set_postfix({
                "loss_D": loss_D_total,
                "loss_G": loss_G_total,
                "loss_md": loss_md_total,
                "loss_cls": loss_cls_total,
                "loss_div": loss_div_total,
            })

            # pbar.set_postfix({
            #     "loss_D": np.mean(loss_D_total),
            #     "loss_G": np.mean(loss_G_total),
            #     "loss_md": np.mean(loss_md_total),
            #     "loss_cls": np.mean(loss_cls_total),
            #     "loss_div": np.mean(loss_div_total),
            # })

        self.scheduler_cgan.step()
        scheduler_server.step()
        
        test_acc = self.evaluate(server, self.testLoader)
        print("[Server | Knowledge Distilaltion]  After test_acc: {:.4f}".format(test_acc))

        # return np.mean(loss_G_total), np.mean(loss_md_total), np.mean(loss_cls_total), np.mean(loss_div_total), np.mean(loss_D_total), avg_acc, server.state_dict().values()
        return loss_G_total, loss_md_total, loss_cls_total, loss_div_total, loss_D_total, avg_acc, server.state_dict().values()


    def train_generator(self, z, y_onehot, labels, generator, student, teacher, weight, num_clients):
        lambda_cls = 1.0
        lambda_div = 1.0

        criterion_cls = nn.CrossEntropyLoss(reduction="mean").to(self.device)
        criterion_diversity = DiversityLoss(metric='l1').to(self.device)

        y = torch.tensor(labels).long().to(self.device)
        fake = generator(z, y_onehot)
        teacher_logit, _ = teacher(fake)
        student_logit, _ = student(fake)

        loss_md = - torch.mean(torch.mean(torch.abs(student_logit - teacher_logit.detach()), dim=1) * weight)
        loss_cls = torch.mean(criterion_cls(teacher_logit, y) * weight.squeeze())
        loss_div = criterion_diversity(z.view(z.shape[0],-1), fake)
        # return loss_md, loss_cls, loss_div

        loss = loss_md + lambda_cls * loss_cls + lambda_div * loss_div / num_clients
        loss.backward()
        return loss, loss_md, loss_cls, loss_div


    def generate_labels(self, number, cls_num):
        labels = np.arange(number)
        proportions = cls_num / cls_num.sum()
        proportions = (np.cumsum(proportions) * number).astype(int)[:-1]
        labels_split = np.split(labels, proportions)
        for i in range(len(labels_split)):
            labels_split[i].fill(i)
        labels = np.concatenate(labels_split)
        np.random.shuffle(labels)
        return labels.astype(int)
    

    def get_batch_weight(self, labels, cls_clnt_weight):
        bs = labels.size
        num_clients = cls_clnt_weight.shape[1]
        batch_weight = np.zeros((bs, num_clients))
        batch_weight[np.arange(bs), :] = cls_clnt_weight[labels, :]
        return batch_weight
    

    def evaluate(self, server_model, testLoader) -> Tuple[float, float]:
        avg_acc = []
        for name, loader in testLoader.items():
            # _, acc_indi, _ = self._test(server_model, loader)
            acc_indi = self._test(server_model, loader)["test_acc"]
            avg_acc.append(acc_indi)
        return np.mean(avg_acc)


class DiversityLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """

    def __init__(self, metric):
        """
        Class initializer.
        """
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """
        Compute the distance between two tensors.
        """
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))