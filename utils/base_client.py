from abc import abstractmethod
from typing import Dict, List, Tuple
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from models.model import MyNet, MyNet_Simple
from models.resnet import ResNet10, ResNet18
from models.mobilenet import MobileNetV2
from models.model_fedgen import FedGenNet
from utils.util import *
from models.model_moon import ModelFedCon


''' Template for client class '''
class BaseClient():
    def __init__(self, device: torch.device, args: ArgumentParser):
        ''' Each client has a model and a dataloader'''
        self.trainLoader = None
        self.valLoader = None

        if args.model == "SimpleCNN":
            self.model = MyNet_Simple(num_classes=args.num_classes).to(device)
        elif args.model == "MyCNN":
            self.model = MyNet(num_classes=args.num_classes).to(device)
        elif args.model == "ResNet10":
            self.model = ResNet10(num_classes=args.num_classes).to(device)
        elif args.model == "FedGenNet":
            self.model = FedGenNet().to(device)
        elif args.model == "MoonNet":
            self.model = ModelFedCon(out_dim=256, n_classes=args.num_classes).to(device)
        elif args.model == "MobileNetV2":
            self.model = MobileNetV2(num_classes=args.num_classes).to(device)
        else:
            raise ValueError("Model `{}` is not defined.".format(args.model))

    def get_parameters(self,):
        ''' Get the parameters of the model '''
        return [val for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: List[np.ndarray]):
        ''' Set the parameters of the model '''
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray]):
        ''' Train the model with the given parameters '''
        self.set_parameters(parameters)
        train_loss, train_acc = self.train()
        return train_loss, train_acc
    
    def evaluate(self, parameters: List[np.ndarray]):
        ''' Evaluate the model with the given parameters '''
        self.set_parameters(parameters)
        val_loss, val_acc = self.test()
        return val_loss, val_acc

    @abstractmethod
    def train(self,):
        pass
    
    @abstractmethod
    def test(self,):
        pass