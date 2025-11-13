from typing import List, Tuple
from .fedavg import FedAvg
import numpy as np
import copy
import torch
from functools import reduce
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.optimizer import required
from collections import OrderedDict



class MIFA():
    def __init__(self, model, device, args,):
        self.args = args
        self.device = device

        self.optimizer = GD(model.parameters(), lr=0.08, weight_decay=0.001)
        self.update_table = self.initialize_mifa(model)  # store previous updates


    def initialize_mifa(self, global_model):
        model_param = self.get_parameters(global_model)
        update_table = [[torch.zeros_like(param) for param in model_param]
                        for _ in range(self.args.num_clients)]
        return update_table


    def update_mifa(self, client, global_model):
        model_param = self.get_parameters(client.model)
        global_param = self.get_parameters(global_model)

        self.update_table[client.cid] =  [1/self.optimizer.get_current_lr() * (local - global_p) 
                                          for local, global_p in zip(model_param, global_param)]
        # self.update_table[client.cid] = 1/self.optimizer.get_current_lr() * (model_param - global_param)


    def aggregation_mifa(self, global_model, rounds):
        global_param = self.get_parameters(global_model)

        # global_param = global_param + self.optimizer.get_current_lr() * sum(self.update_table) / len(self.update_table)
        weights_prime = [
            reduce(torch.add, layer_updates) * self.optimizer.get_current_lr() / len(self.update_table)
            for layer_updates in zip(*self.update_table)
        ]
        weights_prime = [global_p + local_p for global_p, local_p in zip(global_param, weights_prime)]

        self.optimizer.inverse_prop_decay_learning_rate(rounds + 1)
        return weights_prime


    def get_parameters(self, model):
        ''' Get the parameters of the model '''
        return [val for _, val in model.state_dict().items()]


    # def get_flat_params(self, model):
    #     params = []
    #     for param in model.parameters():
    #         params.append(param.data.view(-1))

    #     flat_params = torch.cat(params)
    #     return flat_params.detach()


    # def set_flat_params(self, model, flat_params):
    #     prev_ind = 0
    #     for param in model.parameters():
    #         flat_size = int(np.prod(list(param.size())))
    #         param.data.copy_(
    #             flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
    #         prev_ind += flat_size







class GD(Optimizer):
    def __init__(self, params, lr=required, weight_decay=0):
        self.lr = lr
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(GD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(GD, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha = weight_decay)
                p.data.add_(d_p , alpha = -group['lr'])
        return loss

    def adjust_learning_rate(self, round_i):
        raise BaseException("Deleted.")

        lr = self.lr * (0.5 ** (round_i // 30))
        for param_group in self.param_groups:
            param_group['lr'] = lr

    def soft_decay_learning_rate(self):
        raise BaseException("Deleted.")
        self.lr *= 0.99
        for param_group in self.param_groups:
            param_group['lr'] = self.lr

    def inverse_prop_decay_learning_rate(self, round_i):
        for param_group in self.param_groups:
            param_group['lr'] = self.lr/(round_i+1)

    def set_lr(self, lr):
        raise BaseException("Deleted.")
        for param_group in self.param_groups:
            param_group['lr'] = lr

    def get_current_lr(self):
        return self.param_groups[0]['lr']