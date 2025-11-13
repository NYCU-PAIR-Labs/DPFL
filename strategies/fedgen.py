from typing import List, Tuple, Dict, Any
from .fedavg import FedAvg
from .strategy import Strategy
import numpy as np
import copy
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from utils.loss_fn import DiversityLoss
from models.model import FedGen_Generator

'''
Hyperparameters (From paper):
    - dataset: MNIST (Dirichlet distribution, a = 0.05, 0.1, 1.0)
    - model: -
    - batch: 32
    - lr: 0.01
    - lr_decay: 0.99
    - local_step: 20
    - communication_rounds: 200
    - num_clients: 10, total 20 (sample rate: 0.5)

    'mnist':
        {
            'ensemble_lr': 3e-4,
            'ensemble_batch_size': 128,
            'ensemble_epochs': 50,
            'num_pretrain_iters': 20,
            'ensemble_alpha': 1,    # teacher loss (server side)
            'ensemble_beta': 0,     # adversarial student loss
            'ensemble_eta': 1,      # diversity loss
            'unique_labels': 10,    # available labels
            'generative_alpha': 10, # used to regulate user training
            'generative_beta': 10, # used to regulate user training
            'weight_decay': 1e-2
        },
'''

class FedGen(FedAvg):
    def __init__(self, device, args, params):
        super().__init__(device=device, args=args, params=params)
        ''' Parameters for data-free knowledge distillation '''
        self.ensemble_lr = 3e-4
        self.weight_decay = 1e-2
        self.ensemble_epochs = int(params["ensemble_epochs"])

        self.ensemble_alpha = params["ensemble_alpha"]          # teacher loss (server side)
        self.ensemble_beta = 0                                  # adversarial student loss
        self.ensemble_eta = params["ensemble_eta"]              # diversity loss
        self.unique_labels = args.num_classes                   # available labels
        self.generative_alpha = params["generative_alpha"]      # used to regulate user training
        self.generative_beta = params["generative_alpha"]       # used to regulate user training
        
        self.batch_size = args.batch_size
        self.gen_batch_size = int(params["gen_batch_size"])
        self.available_labels = args.num_classes

        self.ensemble_loss = nn.KLDivLoss(reduction="batchmean")
        self.diversity_loss = DiversityLoss(metric='l1')

        self.generative_model = None
        self.generative_optimizer = None

        self.label_weights = None
        self.qualified_labels = None
        self.z_dim = 32 * 2


    def _initialization(self, **kwargs) -> None:
        self.generator = FedGen_Generator(z_dim=self.z_dim, output_dim=512, num_classes=self.args.num_classes).to(self.device)
        self.optimizer_gen = torch.optim.Adam(self.generator.parameters(), lr=self.ensemble_lr, weight_decay=self.weight_decay)
        self.lr_scheduler_gen = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer_gen, gamma=0.98)


    def _server_train_func(self, cid, rounds, client_list, **kwargs) -> None:
        ''' Train function for the server to orchestrate the training process '''
        result = client_list[cid].train(generator=self.generator, rounds=rounds)
        train_loss, train_acc, teacher_loss, latent_loss = result["train_loss"], result["train_acc"], result["teacher_loss"], result["latent_loss"]
        print("[Client {}] loss: {:.4f}, teacher_loss: {:.4f}, latent_loss: {:.4f}, acc: {:.4f}".format(cid, train_loss, teacher_loss, latent_loss, train_acc))
    

    def _server_agg_func(self, rounds, client_list, active_clients, global_model):
        ''' Aggregate function for the server '''
        new_weights = self._aggregation(client_list)

        self.train_generator(global_model, client_list, active_clients)
        return new_weights
    

    def _train(self, model, trainLoader, optimizer, num_epochs, **kwargs) -> Tuple[float, float]:
        ''' Train function for the client '''
        generator, glob_iter = kwargs["generator"], kwargs["rounds"]

        model.train()
        generator.eval()

        for _ in range(num_epochs):
            total_loss, total_correct = [], []
            total_teacher_loss, total_latent_loss = [], []

            for x, label in trainLoader:
                x, label = x.to(self.device), label.to(self.device)
                output, _ = model(x)

                predict = torch.argmax(output.data, 1)
                total_correct.append((predict == label).sum().item() / len(predict))

                ''' original cross-entropy loss '''
                loss_ce = F.cross_entropy(output, label)

                ''' sample y and generate z '''
                if glob_iter > 1:
                    generative_alpha = self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=self.generative_alpha)
                    generative_beta = self.exp_lr_scheduler(glob_iter, decay=0.98, init_lr=self.generative_beta)

                    ''' get generator output(latent representation) of the same label '''
                    z = torch.randn((label.shape[0], self.z_dim), device=self.device)
                    gen_output = generator(z, label)
                    logit_given_gen = model.classifier(gen_output)
                    user_latent_loss = generative_beta * self.ensemble_loss(F.log_softmax(output, dim=1), F.softmax(logit_given_gen, dim=1))

                    ''' yeilds ideal prediction on the augmented data '''
                    sampled_y = np.random.choice(self.available_labels, self.gen_batch_size)
                    sampled_y = torch.tensor(sampled_y, device=self.device)
                    z = torch.randn((self.gen_batch_size, self.z_dim), device=self.device)
                    gen_output = generator(z, sampled_y)

                    user_output_logp = model.classifier(gen_output)
                    teacher_loss =  generative_alpha * torch.mean(F.cross_entropy(user_output_logp, sampled_y))

                    # this is to further balance oversampled down-sampled synthetic data
                    gen_ratio = self.gen_batch_size / self.batch_size

                    loss = loss_ce + gen_ratio * teacher_loss + user_latent_loss
                    total_teacher_loss.append(teacher_loss.item())
                    total_latent_loss.append(user_latent_loss.item())
                else:
                    loss = loss_ce
                    total_teacher_loss, total_latent_loss = [0], [0]

                total_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)               # clip the gradient to prevent exploding
                optimizer.step()

        # lr_scheduler.step()
        # return np.mean(total_loss), np.mean(total_correct), np.mean(total_teacher_loss), np.mean(total_latent_loss)
        return {
            "train_loss": np.mean(total_loss),
            "train_acc": np.mean(total_correct),
            "teacher_loss": np.mean(total_teacher_loss),
            "latent_loss": np.mean(total_latent_loss)
        }


    def train_generator(self, student_model, client_list, active_client_list):
        self.generator.train()
        student_model.eval()

        self.label_weights, self.qualified_labels = self.get_label_weights(client_list, active_client_list)
        total_teacher_loss, total_student_loss, total_diversity_loss = [], [], []
        
        pbar = tqdm(range(self.ensemble_epochs), desc="[Server | Train Generator]", leave=False)
        for _ in pbar:
            y = np.random.choice(self.qualified_labels, self.batch_size)
            y_input = torch.tensor(y, device=self.device).long()

            ''' feed to generator '''
            z = torch.randn((y.shape[0], self.z_dim), device=self.device)
            gen_output = self.generator(z, y_input)
            
            ''' compute diversity loss '''
            diversity_loss = self.diversity_loss(z, gen_output)             # encourage different outputs

            ''' get teacher loss '''
            teacher_loss = 0
            teacher_logit = 0
            for idx in active_client_list:
                _model = client_list[idx].model
                _model.eval()
            
                weight = self.label_weights[y][:, idx].reshape(-1, 1)
                expand_weight = np.tile(weight, (1, self.unique_labels))

                user_result_given_gen = _model.classifier(gen_output)
                teacher_loss_ = torch.mean(
                    F.cross_entropy(user_result_given_gen, y_input) * \
                    torch.tensor(weight, dtype=torch.float32, device=self.device))
                teacher_loss += teacher_loss_
                teacher_logit += user_result_given_gen * torch.tensor(expand_weight, dtype=torch.float32, device=self.device)

            ''' get student loss '''
            student_output = student_model.classifier(gen_output)
            student_loss = F.kl_div(F.log_softmax(student_output, dim=1), F.softmax(teacher_logit, dim=1), reduction="batchmean")
            
            total_teacher_loss.append(teacher_loss.item() * self.ensemble_alpha)
            total_student_loss.append(student_loss.item() * self.ensemble_beta)
            total_diversity_loss.append(diversity_loss.item() * self.ensemble_eta)
            
            if self.ensemble_beta > 0:
                loss=self.ensemble_alpha * teacher_loss - self.ensemble_beta * student_loss + self.ensemble_eta * diversity_loss
            else:
                loss=self.ensemble_alpha * teacher_loss + self.ensemble_eta * diversity_loss

            self.optimizer_gen.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 5)               # clip the gradient to prevent exploding
            self.optimizer_gen.step()

            pbar.set_postfix({
                "loss_T": np.mean(total_teacher_loss),
                "loss_D": np.mean(total_diversity_loss),
            })

        self.lr_scheduler_gen.step()
        print("[Server | Train Generator] loss_Teacher: {:.4f}, loss_Diversity: {:.4f}".format(np.mean(total_teacher_loss), np.mean(total_diversity_loss)))


    def get_label_weights(self, client_list, active_client_list):
        MIN_SAMPLES_PER_LABEL = 1
        label_weights = np.zeros((self.args.num_classes, len(client_list)))
        for i in active_client_list:
            for _, label in client_list[i].trainLoader.dataset:
                label_weights[label, i] += 1

        qualified_labels = np.where(label_weights.sum(axis=1) >= MIN_SAMPLES_PER_LABEL)[0]
        for i in range(self.args.num_classes):
            if np.sum(label_weights[i], axis=0) > 0:                    # avoid division by zero (lack of data at current round)
                label_weights[i] /= np.sum(label_weights[i], axis=0)
            else:
                label_weights[i] = 0

        label_weights = label_weights.reshape((self.unique_labels, -1))

        # print("label_wieghts:\n", label_weights, label_weights.shape)
        # print("qualified_labels:\n", qualified_labels, qualified_labels.shape)
        # input("Press Enter to continue...")
        return label_weights, qualified_labels
    

    def exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        """Decay learning rate by a factor of 0.95 every lr_decay_epoch epochs."""
        lr= max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr


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


class pFedIBOptimizer(Optimizer):
    def __init__(self, params, lr=0.01):
        # self.local_weight_updated = local_weight # w_i,K
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults=dict(lr=lr)
        super(pFedIBOptimizer, self).__init__(params, defaults)

    def step(self, apply=True, lr=None, allow_unused=False):
        grads = []
        # apply gradient to model.parameters, and return the gradients
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None and allow_unused:
                    continue
                grads.append(p.grad.data)
                if apply:
                    if lr == None:
                        p.data= p.data - group['lr'] * p.grad.data
                    else:
                        p.data=p.data - lr * p.grad.data
        return grads


    def apply_grads(self, grads, beta=None, allow_unused=False):
        #apply gradient to model.parameters
        i = 0
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None and allow_unused:
                    continue
                p.data= p.data - group['lr'] * grads[i] if beta == None else p.data - beta * grads[i]
                i += 1
        return