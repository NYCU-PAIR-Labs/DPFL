from typing import List, Tuple, Dict, Any
from .fedavg import FedAvg
import numpy as np
import copy
from tqdm import tqdm
import torch
from functools import reduce
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset
from utils.data_model import DataModel
from colorama import Fore, Style
import os

'''
Hyperparameters (From paper):
    - dataset: CIFAR10 (public dataset) / CIFAR100 (subset, private dataset)
    - model: 3-layer CNN
    - communication_rounds: 13
    - num_clients: 10

    Pretrain (on public dataset):
        - batch_size: 128
        - epochs: 20

    Fine-tune (on private dataset):
        - batch_size: 32
        - epochs: 25

    Local training:
        - N_alignment: 5000
        - N_logit_matching_round: 1
        - N_private_training_round: 10
        - private_batch_size: 10
        - logit_matching_batch_size: 128
'''

class FedMD(FedAvg):
    def __init__(self, device, args, params):
        super().__init__(device=device, args=args, params=params)
        self.public_train = None
        self.public_test = None
        self.public_size = 1000
        self.public_data_indices = []
        self.public_loader = None
        self.new_public_loader = None

        self.pretrain_epochs = 100                                      # Pretrain epochs to converge on public dataset
        self.finetune_epochs = int(params["finetine_epochs"])           # Fine-tune epochs to converge on private dataset
        self.logit_matching_epochs = int(params["logit_matching_epochs"])

        self.local_payload["type"] = "logits"  # Type of payload (e.g., logits, prototypes)


    def _initialization(self, **kwargs) -> None:
        client_list, active_clients, global_model = kwargs["client_list"], kwargs["active_clients"], kwargs["global_model"]

        self.public_train, self.public_test = self.init_global_dataset()
        self.public_loader_init = DataLoader(self.public_train, batch_size=self.args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        self.valLoader = DataLoader(self.public_test, batch_size=self.args.batch_size)

        total_trainset_num = sum([len(client_list[i].trainLoader.dataset) + len(client_list[i].valLoader.dataset) for i in range(len(client_list))])
        self.public_size = total_trainset_num // 10
        print(Fore.RED + "[Server] Total trainset: {}, public_size: {}".format(total_trainset_num, self.public_size) + Fore.RESET)

        ''' 1. Pretrain on public dataset '''
        if self.args.dataset == "Mnist":
            dt_name = "EMNIST"
        elif self.args.dataset == "Digits" or self.args.dataset == "Cifar10":
            dt_name = "CIFAR100"
        elif self.args.dataset == "Cifar100":
            dt_name = "TinyImageNet"
        elif self.args.dataset == "Office-Caltech":
            dt_name = "Caltech-101"
        else:
            raise ValueError("No supported pretrain dataset for : {}".format(self.args.dataset))
        
        print(Fore.RED + "\n[{}] Trainset: {}, valset: {}".format(dt_name, len(self.public_train), len(self.public_test)) + Fore.RESET)
        try:
            ckpt = torch.load('./checkpoint/fedmd_init_{dt}.pth'.format(dt=self.args.dataset), map_location=self.device)
            global_model.load_state_dict(ckpt['model'])
            public_loss, public_acc = ckpt['loss'], ckpt['acc']
        except:
            public_loss, public_acc = self.pretrain(global_model, self.public_loader_init, self.valLoader, torch.optim.SGD(global_model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay), mode="pretrain")
            os.makedirs("./checkpoint/", exist_ok=True)
            torch.save({
                'model': global_model.state_dict(),
                'loss': public_loss,
                'acc': public_acc,
            }, './checkpoint/fedmd_init_{dt}.pth'.format(dt=self.args.dataset))
        print(Fore.RED + "[Server, Pretrain on {}] loss: {:.4f}, accuracy: {:.4f}".format(dt_name, public_loss, public_acc) + Fore.RESET)

        ''' 2. Fine-tune on private dataset '''
        try:
            for i in range(len(client_list)):
                ckpt_client = torch.load('./checkpoint/fedmd/{dt}/client_{id}_{st}_{a}_{seed}.pth'.format(dt=self.args.dataset, id=i, st=self.args.skew_type, a=self.args.alpha, seed=self.args.seed), map_location=self.device)
                client_list[i].model.load_state_dict(ckpt_client['model'])
                private_loss, private_acc = ckpt_client['loss'], ckpt_client['acc']
                print(Fore.RED + "[Client {}, Fine-tune on private set] loss: {:.4f}, accuracy: {:.4f}".format(i, private_loss, private_acc) + Fore.RESET)
        except:
            os.makedirs("./checkpoint/fedmd/{dt}/".format(dt=self.args.dataset), exist_ok=True)
            for i in range(len(client_list)):
                # client_list[i].set_parameters(self.get_parameters())
                private_loss, private_acc = client_list[i].strategy.pretrain(client_list[i].model, client_list[i].trainLoader, client_list[i].valLoader, client_list[i].optimizer, mode="fine-tune")
                torch.save({
                    'model': client_list[i].model.state_dict(),
                    'loss': private_loss,
                    'acc': private_acc,
                }, './checkpoint/fedmd/{dt}/client_{id}_{st}_{a}_{seed}.pth'.format(dt=self.args.dataset, id=i, st=self.args.skew_type, a=self.args.alpha, seed=self.args.seed))
                print(Fore.RED + "[Client {}, Fine-tune on private set] loss: {:.4f}, accuracy: {:.4f}".format(i, private_loss, private_acc) + Fore.RESET)

        # ''' 3. Get first global dataset & global logits '''
        # self.strategy.public_loader, self.strategy.global_payload = self.strategy.generate_new_global_logits(self.client_list, self.active_clients, rounds=0)
        self.new_public_loader = self.get_public_data(rounds=0)


    def _server_train_func(self, cid, rounds, client_list, **kwargs) -> None:
        ''' Train function for the server to orchestrate the training process '''
        result = client_list[cid].train(public_loader=self.public_loader, new_public_loader=self.new_public_loader, global_logits=kwargs["global_payload"])
        train_loss, train_acc, logit_loss = result["train_loss"], result["train_acc"], result["logit_loss"]
        print("[Client {}] loss: {:.4f}, logit_loss: {:.4f}, acc: {:.4f}".format(cid, train_loss, logit_loss, train_acc))
    

    def _server_agg_func(self, rounds, client_list, active_clients, global_model):
        ''' Aggregate function for the server '''
        new_weights = self._aggregation(client_list)
        self.global_payload = self.payload_aggregation(
            local_payload_list = [client.strategy.local_payload["data"] for client in client_list if client.cid in active_clients],
            client_weight_list = [1 / len(active_clients) for _ in range(len(active_clients))],
        )

        self.public_loader = self.new_public_loader
        self.new_public_loader = self.get_public_data(rounds)
        return new_weights


    def init_global_dataset(self,):
        ''' 
            - CIFAR-100
            source_classes = [
                "rocket", "bus", "butterfly", "fox", "camel",
                "wolf", "lizard", "cattle", "seal", "pickup_truck"
            ]
            source_classes = [69, 13, 14, 34, 15, 97, 44, 19, 72, 58]

            target_classes = [
                'airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck'
            ]
        
            - Caltech-101
            source_classes = [
                "stapler", "Motorbikes", "chair", "cellphone", "camera",
                "laptop", "watch", "scissors", "lamp", "windsor_chair"
            ]
            source_classes = [85, 3, 22, 21, 17, 57, 94, 80, 56, 98]

            target_classes = [
                "back_pack", "bike", "calculator", "headphone", "keyboard",
                "laptop_computer", "monitor", "mouse", "mug", "projector"
            ]
        '''

        if self.args.dataset == "Mnist":
            dataset, testset = DataModel().load_data(dataset_name="EMNIST", args=self.args)
            source_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        
        elif self.args.dataset == "Digits" or self.args.dataset == "Cifar10":
            dataset, testset = DataModel().load_data(dataset_name="CIFAR100", args=self.args)
            source_classes = [69, 13, 14, 34, 15, 97, 44, 19, 72, 58]

        elif self.args.dataset == "Cifar100":
            dataset, testset = DataModel().load_data(dataset_name="TinyImageNet", args=self.args)

            # Selected 100 classes similar to CIFAR-100
            source_classes = [i for i in range(100)]  # All classes from 0 to 99

        elif self.args.dataset == "Office-Caltech":
            dataset, testset = DataModel().load_data(dataset_name="Caltech101", args=self.args)
            source_classes = [85, 3, 22, 21, 17, 57, 94, 80, 56, 98]

            # print("Public dataset size: ", len(dataset))
            # print("Categories: ", dataset.categories)
            # for i, category in enumerate(dataset.categories):
            #     if category in source_classes:
            #         print(f"{i}: {category}")

        # Get the indices of samples belonging to the desired subclasses
        train_indices = []
        for i, (_, label) in enumerate(dataset):
            if label in source_classes:
                train_indices.append(i)

        test_indices = []
        for i, (_, label) in enumerate(testset):
            if label in source_classes:
                test_indices.append(i)

        # Create a Subset dataset containing samples from the desired subclasses
        label_mapping = {source_classes[i]: i for i in range(len(source_classes))}
        self.public_train = RelabelDataset(Subset(dataset, train_indices), label_mapping)
        self.public_test = RelabelDataset(Subset(testset, test_indices), label_mapping)
        
        ''' initialize the public dataset indices for every round '''
        for i in range(self.args.num_rounds):
            self.public_data_indices.append(np.random.choice(len(self.public_train), self.public_size, replace=False))
        
        return self.public_train, self.public_test


    def _train(self, model, trainLoader, optimizer, num_epochs, **kwargs) -> Tuple[float, float, float]:
        ''' Train function for the client '''
        public_loader = kwargs["public_loader"]
        new_public_loader = kwargs["new_public_loader"]
        global_logits = kwargs["global_logits"]

        model.train()

        # divide the global logits into batches for alignment
        if public_loader is not None:
            global_logits = torch.split(global_logits, self.args.batch_size)
            for _ in range(self.logit_matching_epochs):
                logit_loss = []
                if len(global_logits) > 0:
                    for (x, label), logit_g in zip(public_loader, global_logits):
                        x, label, logit_g = x.to(self.device), label.to(self.device), logit_g.to(self.device)
                        logit_l, _ = model(x)

                        # Logit loss
                        loss = F.l1_loss(logit_l, logit_g)
                        logit_loss.append(loss.item())
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)               # clip the gradient to prevent exploding
                        optimizer.step()
        else:
            logit_loss = [0]
            
        for i in range(num_epochs):
            total_loss, total_correct = [], []
            for x, label in trainLoader:
                x, label = x.to(self.device), label.to(self.device)
                output, _ = model(x)

                predict = torch.argmax(output.data, 1)
                total_correct.append((predict == label).sum().item() / len(predict))

                # Cross entropy loss
                loss = F.cross_entropy(output, label)
                total_loss.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)               # clip the gradient to prevent exploding
                optimizer.step()

        self.local_payload["data"] = self.calculate_logits(model, new_public_loader)
        return {
            "train_loss": np.mean(total_loss),
            "train_acc": np.mean(total_correct),
            "logit_loss": np.mean(logit_loss)
        }
    

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


    def pretrain(self, model, loader, valLoader, optimizer, mode="pretrain"):
        epochs = self.pretrain_epochs if mode == "pretrain" else self.finetune_epochs
        best_model = copy.deepcopy(model.state_dict())
        best_acc, best_loss = 0, 0

        t = tqdm(range(epochs), desc=mode, leave=False)
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

            result = self._test(model, valLoader)
            test_loss, test_acc = result["test_loss"], result["test_acc"]
            if test_acc > best_acc:
                best_acc = test_acc
                best_loss = test_loss
                best_model = copy.deepcopy(model.state_dict())
            
            t.set_postfix(loss="{:.4f}".format(np.mean(total_loss)), accuracy="{:.4f}".format(np.mean(total_correct)), test_loss="{:.4f}".format(test_loss), test_acc="{:.4f}".format(test_acc))
        model.load_state_dict(best_model)
        return best_loss, best_acc


    def payload_aggregation(self, local_payload_list, client_weight_list):
        """ Aggregate the local logits from clients """
        if len(local_payload_list) == 0:
            return None
        
        # Weighted average of local logits
        weighted_logits = [local_payload_list[i] * client_weight_list[i] for i in range(len(local_payload_list))]
        aggregated_logits = reduce(lambda x, y: x + y, weighted_logits)
        aggregated_logits /= sum(client_weight_list)
        return aggregated_logits


    def generate_new_global_logits(self, client_list, active_client_list, rounds):
        public_loader = self.get_public_data(rounds)
        global_logits = self.calculate_global_logits(public_loader, client_list, active_client_list)
        return public_loader, global_logits


    def calculate_global_logits(self, dataLoader, client_list, active_client_list):
        global_logits = []
        for i in active_client_list:
            local_logit = self.calculate_logits(client_list[i].model, dataLoader)
            global_logits.append(local_logit)
        global_logits = torch.mean(torch.stack(global_logits), dim=0)
        return global_logits
    

    def calculate_logits(self, model, dataLoader):
        """ Calculate the global logits """
        model.eval()
        with torch.no_grad():
            logits = []
            for x, labels in dataLoader:
                x, labels = x.to(self.device), labels.to(self.device)
                output, _ = model(x)
                logits.append(output)

        logits = torch.cat(logits, dim=0)
        return logits
    

    def get_public_data(self, rounds):
        ''' Sample 5000 data from public dataset for knowledge distillation '''
        public_subset = torch.utils.data.Subset(self.public_train, self.public_data_indices[rounds-1])
        loader = torch.utils.data.DataLoader(public_subset, batch_size=self.args.batch_size, shuffle=True, num_workers=0)
        return loader
    

    def _get_local_payload(self) -> Dict[str, Any]:
        """
        Get the current payload from the strategy.
        """
        return self.local_payload


# Define a custom dataset class to modify labels
class RelabelDataset(Dataset):
    def __init__(self, dataset, labels_mapping):
        self.dataset = dataset
        self.labels_mapping = labels_mapping

    def __getitem__(self, index):
        image, label = self.dataset[index]
        relabeled_label = self.labels_mapping[label]
        return image, relabeled_label

    def __len__(self):
        return len(self.dataset)
    

selected_wnids = [
    'n01443537', 'n01641577', 'n01698640', 'n01742172', 'n01768244', 'n01855672',
    'n01882714', 'n01910747', 'n01944390', 'n01945685', 'n01950731', 'n02002724',
    'n02056570', 'n02058221', 'n02074367', 'n02085620', 'n02094433', 'n02099601',
    'n02099712', 'n02106662', 'n02113799', 'n02123045', 'n02129165', 'n02132136',
    'n02165456', 'n02190166', 'n02226429', 'n02233338', 'n02236044', 'n02268443',
    'n02279972', 'n02321529', 'n02364673', 'n02395406', 'n02403003', 'n02415577',
    'n02481823', 'n02504458', 'n02509815', 'n02606052', 'n02655020', 'n02701002',
    'n02747177', 'n02795169', 'n02802426', 'n02808440', 'n02814533', 'n02814860',
    'n02815834', 'n02823428', 'n02837789', 'n02841315', 'n02843684', 'n02892201',
    'n02906734', 'n02948072', 'n02950826', 'n02979186', 'n02988304', 'n03014705',
    'n03026506', 'n03042490', 'n03085013', 'n03100240', 'n03127925', 'n03160309',
    'n03201208', 'n03250847', 'n03255030', 'n03344393', 'n03388043', 'n03417042',
    'n03424325', 'n03444034', 'n03445777', 'n03481172', 'n03538406', 'n03544143',
    'n03584254', 'n03599486', 'n03649909', 'n03666591', 'n03670208', 'n03706229',
    'n03733131', 'n03763968', 'n03770439', 'n03792782', 'n03793489', 'n03838899',
    'n03840681', 'n03877845', 'n03888257', 'n03929660', 'n03942813', 'n03977966',
    'n03995372', 'n04008634', 'n04067472', 'n04099969', 'n04146614', 'n04266014'
]
