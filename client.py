from typing import Dict, List
from colorama import Fore, Style
import importlib
import numpy as np
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from utils.base_client import BaseClient
from strategies.strategy import Strategy
from utils.data_model import DataModel
from utils.util import *

''' FL Client class '''
class FLclient(BaseClient):
    def __init__(self, cid: int, dataset: Dict[str, Dataset], strategy: Strategy, device: torch.device, args: Dict, sim=None):
        super().__init__(device, args)
        '''
            cid: client id
            trainLoader: dataloader for training
            valLoader: dataloader for validation
            _task_id: current task_id if is class-incremental learning
            strategy: strategy (algorithm) for training
            local_protos: prototypes of each class
            device: cpu or gpu
            args: arguments
        '''
        self.args = args
        self.cid = cid
        self._current_tid = 0
        self.dataset = self.init_dataset(dataset)
        self.trainLoader, self.valLoader = None, None
        self.data_distribution = None

        if args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        elif args.optimizer == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise ValueError("Non-supported optimizer")

        # define download/upload/computation latency (seconds)
        # self.dl_latency = np.random.normal(10.0, 3.0)       # depends on bandwidth
        # self.up_latency = np.random.normal(15.0, 3.0)       # depends on bandwidth
        # self.cp_latency = np.random.normal(30.0, 1.0)       # depends on device, data size

        self.status = "online"
        self.strategy = copy.deepcopy(strategy)
        self._strategy = strategy.__class__.__name__
        self.device = device
        self.sim = sim

        self.payload = {                # client's potential payload (logits, prototypes, etc.)
            "type": "None",
            "data": None,
        }

        # save_image(make_grid(next(iter(self.trainLoader))[0][:8], padding=1), 
        #            "{}/cid_{}.png".format(args.figure_path, cid))


    def init_dataset(self, dataset):
        if self.args.incremental_type == "class-incremental":
            tmp_dataset = {}
            for name, ds in dataset.items():
                tmp_dataset[name] = DataModel().divide_dataset_to_incremental_partition(ds, self.args)

            task_dataset = {
                f"task_{i}": {
                    name: tmp_dataset[name][f"task_{i}"] for name in dataset.keys()
                }
                for i in range(1, 6)
            }
            return task_dataset
        else:
            return dataset


    def get_current_dataLoader(self,):
        if self.args.incremental_type == "class-incremental":
            dataset = self.dataset[f"task_{self._current_tid}"]
        else:
            dataset = self.dataset
    
        Dataset = ConcatDataset([dataset[name] for name in dataset])
        val_len = int(len(Dataset) * self.args.val_ratio)
        trainset, valset = random_split(Dataset, [len(Dataset) - val_len, val_len], torch.Generator().manual_seed(self.args.seed + self.cid))

        trainLoader = DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True, drop_last=False, num_workers=2, pin_memory=True)
        valLoader = DataLoader(valset, batch_size=self.args.batch_size)
        return trainLoader, valLoader
    

    def update_incremental_data(self,):
        ''' Update the client's data if is under class-incremental learning '''
        self._current_tid += 1
        self.trainLoader, self.valLoader = self.get_current_dataLoader()
        self.data_distribution = self.get_data_distribution()

    
    def get_data_distribution(self,):
        result = {}
        for _, label in self.trainLoader:
            for l in label:
                if l.item() in result:
                    result[l.item()] += 1
                else:
                    result[l.item()] = 1
        sorted_result = dict(sorted(result.items(), key=lambda x: x[0]))
        return sorted_result
    

    def reset_latency(self,):
        ''' Set the client's latency '''
        self.dl_latency = np.random.normal(10.0, 3.0)                                 # depends on bandwidth
        self.up_latency = np.random.normal(15.0, 3.0)                                 # depends on bandwidth``
        self.cp_latency = np.random.normal(30.0, 1.0)                                 # depends on device, data size


    ''' Train function for the client '''
    def train(self, **kwargs):  
        self.model.to(self.device)                  # move model to gpu

        # Log the virtual time when training starts
        if self.sim:
            self.sim.sleep(self.dl_latency)  # Simulate model downloading time
            print(f"{Fore.BLACK}{format_sim_time(self.sim.now)} |{Fore.RESET} [Client {self.cid}] downloaded model from server")
                  
            self.sim.sleep(self.cp_latency)   # Simulate compute latency *before* real [training
            print(f"{Fore.BLACK}{format_sim_time(self.sim.now)} |{Fore.RESET} [Client {self.cid}] trained model")

        # Run actual training (takes real wall time, not virtual time)
        result = self.strategy._train(self.model, self.trainLoader, self.optimizer, self.args.num_epochs, **kwargs)
        self.local_payload = self.strategy._get_local_payload()

        if self.sim:
            self.sim.sleep(self.up_latency)    # Simulate upload time *after* training is truly done
            print(f"{Fore.BLACK}{format_sim_time(self.sim.now)} |{Fore.RESET} [Client {self.cid}] uploaded model to server")

        self.model.to("cpu")                   # move model back to cpu to save memory
        # return self.strategy._train(self.model, self.trainLoader, self.optimizer, self.args.num_epochs, **kwargs)
        return result


    ''' Test function for the client '''
    def test(self, **kwargs):
        self.model.to(self.device)
        result = self.strategy._test(self.model, self.valLoader, **kwargs)
        self.model.to("cpu")
        return result