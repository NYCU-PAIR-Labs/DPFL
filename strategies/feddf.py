from typing import List, Tuple, Dict, Any
import torch.utils
from .fedavg import FedAvg
import numpy as np
import copy
import torch
from torch import nn
from functools import reduce
from tqdm.auto import tqdm
import torch.nn.functional as F
from utils.loss_fn import DiversityLoss
from utils.data_model import load_public_data
from utils.util import *

'''
Hyperparameters (From paper):
    - dataset: CIFAR-10 & CIFAR-100 (Distillation dataset)

    Local training:
        - model: ResNet-8
        - optimizer: SGD
        - lr: 0.1 (ResNet-like nets), no lr_decay, no momentum, no weight_decay
        - lr_scheduler: MultiStepLR
        - lr_decay: 0.1
        - batch_size: 64
        
    Model fusion (distillation):
        - optimizer: Adam
        - lr: 0.001, with cosine annealing
        - early stopping (stop distillation after the validation performance plateaus for 1000 steps / total 10000 steps)
        - batch_size: 128
    
    - local_epoch: 40
    - communication_rounds: 100
    - num_clients: total 20 (sampling fraction: 0.2, 0.4, 0.8)
'''

class FedDF(FedAvg):
    def __init__(self, device, args, params):
        super().__init__(device=device, args=args, params=params)
        self.dataLoader = None          # dataloader for unlabeled dataset
        self.valLoader = None

        ''' Hyperparameters for ensemble distillation (unlabeled dataset) '''
        self.ensemble_epoch = int(params["ensemble_epoch"])
        self.sever_local_steps = int(params["sever_local_steps"])
        self.T = params["T"]
        self.d_batch_size = int(params["d_batch_size"])

        # self.ensemble_alpha = 1         # teacher loss (server side)
        # self.ensemble_beta = 0          # adversarial student loss
        # self.ensemble_eta = 1           # diversity loss
        # self.ensemble_samples = 1000

        ''' Hyperparameters for ensemble distillation (generator) '''
        # self.generator = None
        # self.optimizer_gen = None
        # self.lr_scheduler_gen = None

        # self.num_classes = args.num_classes
        # self.label_weights = None
        # self.qualified_labels = None
        # self.unique_labels = 10         # available labels
        # self.z_dim = 100

        # self.generator_epoch = 20
        # self.gen_batch_size = args.batch_size
        # self.available_labels = args.num_classes

        self.diversity_loss = DiversityLoss(metric='l1')


    def _initialization(self, **kwargs) -> None:
        client_list = kwargs["client_list"]

        ''' Intialize unlabeled dataset for ensemble distillation (does not work well) '''
        unlabeled_dataset = load_public_data(dataset=self.args.dataset, args=self.args)
        self.dataLoader = DataLoader(unlabeled_dataset, batch_size=self.d_batch_size, shuffle=True, num_workers=4, pin_memory=True)

        ''' Initialize the generator for ensemble distillation '''
        # self.strategy.generator = models.CGenerator().to(self.device)
        # self.strategy.optimizer_gen = torch.optim.Adam(self.strategy.generator.parameters(), lr=0.01)
        # self.strategy.lr_scheduler_gen = torch.optim.lr_scheduler.StepLR(self.strategy.optimizer_gen, step_size=1, gamma=0.998)

        ''' Get client's validation data '''
        valset = [client_list[i].valLoader.dataset for i in range(len(client_list))]
        self.valLoader = {
            "total_val": DataLoader(ConcatDataset(valset), batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
        }


    def _server_train_func(self, cid, rounds, client_list, **kwargs) -> None:
        ''' Train function for the server to orchestrate the training process '''
        result = client_list[cid].train(**kwargs)
        train_loss, train_acc = result["train_loss"], result["train_acc"]
        print("[Client {}] loss: {:.4f}, acc: {:.4f}".format(cid, train_loss, train_acc))
    

    def _server_agg_func(self, rounds, client_list, active_clients, global_model):
        ''' Aggregate function for the server '''
        new_weights = self._aggregation(client_list)
        # self.set_parameters(new_weights)                                    # set global model's weight for distillation
        set_param(global_model, new_weights)                                  # set global model's weight for distillation

        # dis_loss, new_weights = self.strategy.ensemble_distillation(self.model, self.strategy.generator, self.client_list, self.active_clients)
        dis_loss, new_weights = self.ensemble_distillation(global_model, client_list, active_clients)
        print("[Server] ensemble_distillation loss: {:.4f}".format(dis_loss))

        return new_weights



    def _train(self, model, trainLoader, optimizer, num_epochs, **kwargs) -> Tuple[float, float]:
        ''' Train function for the client '''
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

        # return np.mean(total_loss), np.mean(total_correct)
        return {
            "train_loss": np.mean(total_loss),
            "train_acc": np.mean(total_correct)
        }


    def ensemble_distillation(self, global_model, client_list, active_client_list):
        optimizer_server = torch.optim.SGD(global_model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=1e-5)

        intial_val_acc = self.evaluate(global_model, self.valLoader)      # inital val_acc
        state = {
            "best_val_acc": 0,
            "best_server_model": copy.deepcopy(global_model.state_dict()),
            "best_epoch": 0
        }
        total_loss = []

        print("\n[Server | E. Distillation] Start ensemble distillation, inital val_acc: {:.4f}".format(intial_val_acc))
        pbar = tqdm(range(self.ensemble_epoch), desc="[Server | E. Distillation]", leave=False)
        
        global_model.train()
        for n in pbar:
            try:
                x, _ = next(self.data_iter)
            except:
                self.data_iter = iter(self.dataLoader)
                x, _ = next(self.data_iter)
            x = x.to(self.device)

            # teacher (clients) logits
            with torch.no_grad():
                teacher_logits = []
                for i in active_client_list:
                    _teacher = client_list[i].model
                    _teacher.eval()
                    logits, _ = _teacher(x)
                    teacher_logits.append(logits)
                teacher_logits = torch.mean(torch.stack(teacher_logits), dim=0)

            for _ in range(self.sever_local_steps):
                # student (server) logits
                student_logits, _ = global_model(x)

                # loss
                loss = self.KL_loss(student_logits, teacher_logits)
                total_loss.append(loss.item())

                optimizer_server.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(global_model.parameters(), 10)
                optimizer_server.step()

            val_acc = self.evaluate(global_model, self.valLoader)
            pbar.set_postfix({"val_acc": val_acc})
            if val_acc > state["best_val_acc"]:
                state["best_val_acc"] = val_acc
                state["best_server_model"] = copy.deepcopy(global_model.state_dict())

        # restore the best model
        global_model.load_state_dict(state["best_server_model"])
        print("[Server | E. Distillation] Best val_acc: {:.4f}".format(state["best_val_acc"]))
        return np.mean(total_loss), global_model.state_dict().values()


    # def get_new_unlabel_data(self, dataset, num_samples):
    #     new_public_subset = torch.utils.data.Subset(dataset, np.random.choice(len(dataset), num_samples, replace=False))
    #     loader = torch.utils.data.DataLoader(new_public_subset, batch_size=self.args.batch_size, shuffle=True, num_workers=2)
    #     return loader


    # ---------- Ensemble distillation with generator ----------
    # def ensemble_distillation(self, global_model, generator, client_list, active_client_list):
    #     ''' First step: train the generator to generate pseudo data '''
    #     self.train_generator(generator, client_list, active_client_list)

    #     optimizer_server = torch.optim.SGD(global_model.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=1e-5)
    #     lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_server, step_size=1, gamma=0.998)

    #     intial_val_acc = self.evaluate(global_model, self.valLoader)      # inital val_acc
    #     state = {
    #         "best_val_acc": 0,
    #         "best_server_model": copy.deepcopy(global_model.state_dict()),
    #     }
    #     total_loss = []

    #     print("\n[Server | E. Distillation] Start ensemble distillation, inital val_acc: {:.4f}".format(intial_val_acc))
    #     pbar = tqdm(range(self.ensemble_epoch), desc="[Server | E. Distillation]", leave=False)
        
    #     global_model.train()
    #     for n in pbar:
    #         sampled_y = np.random.choice(self.available_labels, self.gen_batch_size)
    #         sampled_y = F.one_hot(torch.Tensor(sampled_y).long(), num_classes=self.num_classes).float().to(self.device)
            
    #         # sampled_y = torch.tensor(sampled_y, device=self.device)
    #         z = torch.randn((self.gen_batch_size, self.z_dim, 1, 1), device=self.device)
    #         x = generator(z, sampled_y)

    #         # teacher (clients) logits
    #         with torch.no_grad():
    #             teacher_logits = []
    #             for i in active_client_list:
    #                 _teacher = client_list[i].model
    #                 _teacher.eval()
    #                 logits, _ = _teacher(x)
    #                 teacher_logits.append(logits)
    #             teacher_logits = torch.mean(torch.stack(teacher_logits), dim=0)


    #         for _ in range(self.sever_local_steps):
    #             # student (server) logits
    #             student_logits, _ = global_model(x)

    #             # loss
    #             loss = self.KL_loss(student_logits, teacher_logits)
    #             total_loss.append(loss.item())

    #             optimizer_server.zero_grad()
    #             loss.backward()
    #             # torch.nn.utils.clip_grad_norm_(global_model.parameters(), 5)
    #             optimizer_server.step()
            
    #         # learning rate scheduler
    #         lr_scheduler.step()

    #         # evaluate
    #         val_acc = self.evaluate(global_model, self.valLoader)
    #         pbar.set_postfix({"val_acc": val_acc})

    #         if val_acc > state["best_val_acc"]:
    #             state["best_val_acc"] = val_acc
    #             state["best_server_model"] = copy.deepcopy(global_model.state_dict())

    #     # restore the best model
    #     global_model.load_state_dict(state["best_server_model"])
    #     print("[Server | E. Distillation] Best val_acc: {:.4f}".format(state["best_val_acc"]))
    #     return np.mean(total_loss), global_model.state_dict().values()


    # def train_generator(self, generator, client_list, active_client_list):
    #     self.label_weights, self.qualified_labels = self.get_label_weights(client_list, active_client_list)
    #     total_teacher_loss, total_diversity_loss = [], []
    #     generator.train()
        
    #     pbar = tqdm(range(self.generator_epoch), desc="[Server | Train Generator]", leave=False)
    #     for _ in pbar:
    #         y = np.random.choice(self.qualified_labels, self.args.batch_size)
    #         y_input = F.one_hot(torch.Tensor(y).long(), num_classes=self.num_classes).float().to(self.device)

    #         ''' feed to generator '''
    #         z = torch.randn((y.shape[0], self.z_dim, 1, 1), device=self.device)
    #         gen_output = generator(z, y_input)
            
    #         ''' compute diversity loss '''
    #         diversity_loss = self.diversity_loss(z.view(z.shape[0],-1), gen_output)             # encourage different outputs

    #         ''' get teacher loss '''
    #         teacher_loss = 0
    #         for idx in active_client_list:
    #             _model = client_list[idx].model
    #             _model.eval()

    #             user_result_given_gen, _ = _model(gen_output)
    #             teacher_loss_ = F.cross_entropy(user_result_given_gen, y_input) / len(active_client_list)
    #             teacher_loss += teacher_loss_
           
    #         total_teacher_loss.append(teacher_loss.item() * self.ensemble_alpha)
    #         total_diversity_loss.append(diversity_loss.item() * self.ensemble_eta)

    #         loss = self.ensemble_alpha * teacher_loss + self.ensemble_eta * diversity_loss

    #         self.optimizer_gen.zero_grad()
    #         loss.backward()
    #         self.optimizer_gen.step()

    #         pbar.set_postfix({
    #             "loss_T": np.mean(total_teacher_loss),
    #             "loss_D": np.mean(total_diversity_loss),
    #         })

    #     self.lr_scheduler_gen.step()
    #     print("[Server | Train Generator] loss_Teacher: {:.4f}, loss_Diversity: {:.4f}".format(np.mean(total_teacher_loss), np.mean(total_diversity_loss)))


    # def get_label_weights(self, client_list, active_client_list):
    #     MIN_SAMPLES_PER_LABEL = 1
    #     label_weights = np.zeros((self.args.num_classes, len(client_list)))
    #     for i in active_client_list:
    #         for _, label in client_list[i].trainLoader.dataset:
    #             label_weights[label, i] += 1

    #     qualified_labels = np.where(label_weights.sum(axis=1) >= MIN_SAMPLES_PER_LABEL)[0]
    #     for i in range(self.args.num_classes):
    #         label_weights[i] /= np.sum(label_weights[i], axis=0)

    #     label_weights = label_weights.reshape((self.unique_labels, -1))

    #     # print("label_wieghts:\n", label_weights, label_weights.shape)
    #     # print("qualified_labels:\n", qualified_labels, qualified_labels.shape)
    #     # input("Press Enter to continue...")
    #     return label_weights, qualified_labels


    def KL_loss(self, student_logits, teacher_logits):
        divergence = F.kl_div(
            F.log_softmax(student_logits / self.T, dim=1),
            F.softmax(teacher_logits / self.T, dim=1),
            reduction="batchmean",
        )
        return divergence


    def evaluate(self, server_model, testLoader) -> Tuple[float, float]:
        avg_acc = []
        for name, loader in testLoader.items():
            # _, acc_indi, _ = self._test(server_model, loader)
            acc_indi = self._test(server_model, loader)["test_acc"]
            avg_acc.append(acc_indi)
        return np.mean(avg_acc)


    def _test(self, model, testLoader, **kwargs) -> Tuple[float, float]:
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