from typing import List, Tuple
from colorama import Fore, Style
from .fedavg import FedAvg
import numpy as np
import math
import copy
from tqdm import tqdm
import torch
from torch import nn
from torchvision.utils import make_grid, save_image
import torch.nn.functional as F
from functools import reduce
from torch.utils.data import DataLoader, ConcatDataset
from models.model import CGenerator
from utils.loss_fn import DiversityLoss
np.set_printoptions(suppress=True, linewidth=np.inf)

'''
Hyperparameters:
    - dataset: MNIST, CIFAR-10, Office-Caltech
'''

class FedDPFL(FedAvg):
    def __init__(self, fl_strategy, device, args, params):
        super().__init__(device=device, args=args, params=params)
        self.fl_strategy = fl_strategy
        ''' Parameters for data-free knowledge distillation '''
        self.z_dim = 100
        self.cgan = CGenerator(nz=self.z_dim, ngf=16, img_size=32, n_cls=args.num_classes).to(device)
        self.optimizer_cgan = torch.optim.Adam(self.cgan.parameters(), lr=3e-4, weight_decay=1e-2)
        # self.scheduler_cgan = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_cgan, gamma=0.98)

        ''' Hyperparameters for knowledge distillation'''
        self.batch_size = args.batch_size
        self.gen_batch_size = 128                           # int(params['gen_batch_size'])
        self.iterations = 1                                 # int(params['iterations'])
        self.inner_round_g = int(params['inner_round_g'])
        self.inner_round_d = int(params['inner_round_d'])
        self.T = params['con_T']

        ''' Coefficients for individual loss '''
        self.ensemble_gamma = params['kd_gamma']
        self.ensemble_beta = params['kd_beta']
        self.ensemble_eta = params['kd_eta']
        self.age_ld = 1                                     # params['age_ld']
        self.impt_ld = 1                                    # params['impt_ld']
        self.online_rate = params['online_rate']
        self.offline_rate = params['offline_rate']
        
        ''' Parameters for client data statistics '''
        self.num_classes = args.num_classes
        self.label_weights = []
        self.qualified_labels = []
    
        self.criterion_diversity = DiversityLoss(metric='l1').to(device)
        self.cosine_sim = torch.nn.CosineSimilarity(dim=-1)
        self.valLoader = None
        self.params = params

        '''Knowledge Pool 
            - {
                signature (data_classes): {
                    "model": model,
                    "data_summary": {
                        "label": num_data
                    },
                    "num_data": len(data),
                    "online_age": rounds,
                    "offline_age": rounds,
                }
            }
        '''
        self.knowledge_pool = {}


    def initialization(self, global_model, client_list, testLoader):
        ''' Create lr_scheduler for global model '''
        self.optimizer_server = torch.optim.SGD(global_model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.scheduler_server = torch.optim.lr_scheduler.StepLR(self.optimizer_server, step_size=1, gamma=0.998)

        self.testLoader = copy.deepcopy(testLoader)

        ''' Get client's validation data '''
        valset = [client_list[i].valLoader.dataset for i in range(len(client_list))]
        self.valLoader = {
            "total_val": DataLoader(ConcatDataset(valset), batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
        }
        print(Fore.YELLOW + "\n[Server] Initializing the knowledge pool..." + Fore.RESET)
        print(Fore.YELLOW + "Optimized Parameters for DPFL: {}".format(self.params) + Fore.RESET)


    def data_free_knowledge_distillation(self, global_model, payloads, rounds):
        generator = self.cgan
        self.label_weights, self.qualified_labels = self.get_label_weights_from_knowledge_pool(self.knowledge_pool)
        age_weights = self.get_age_weight()

        ''' Integrate the label weight and age weight '''
        self.label_weights = self.combine_weights(self.label_weights, age_weights * self.age_ld)
        total_label_weights = np.sum(self.label_weights, axis=1)                    # get each label's total weight from clients in knowledge pool

        intial_val_acc = self.evaluate(global_model, self.valLoader, payloads)      # inital val_acc
        state = {
            "best_val_acc": intial_val_acc,
            "best_server_model": copy.deepcopy(global_model.state_dict()),
            "best_generator": copy.deepcopy(generator.state_dict()),
        }
        intial_test_acc = self.evaluate(global_model, self.testLoader, payloads)      # inital test_acc
        print("[Server | E. Distillation] Start ensemble distillation, inital test_acc: {:.4f}".format(intial_test_acc))
        
        pbar = tqdm(range(self.iterations), desc="[Server | E. Knowledge Distillation]", leave=False)
        for _ in pbar:
            ''' Train Generator '''
            generator.train()
            global_model.eval()

            loss_G_total = []
            loss_KD_total = []
            loss_con_total = []
            loss_cls_total = []
            loss_div_total = []

            # y = np.random.choice(self.qualified_labels, self.batch_size)
            y = self.generate_labels(self.batch_size, total_label_weights)
            y_input = F.one_hot(torch.tensor(y), num_classes=self.args.num_classes).type(torch.float32).to(self.device)

            for _ in range(self.inner_round_g):
                ''' feed to generator '''
                z = torch.randn((self.batch_size, self.z_dim), device=self.device)
                gen_output = generator(z, y_input)

                ''' get the student logit '''
                _, student_feature = global_model(gen_output)

                ''' compute diversity loss '''
                loss_div = self.criterion_diversity(z, gen_output)


                ''' Train the generator using contrastive learning to separate the features (class separation) '''
                # create the class feature
                class_features = [[] for _ in range(self.num_classes)]
                for i in range(self.batch_size):
                    class_features[y[i]].append(student_feature[i])
                class_features = [torch.stack(class_feature) if len(class_feature) > 0 else torch.zeros((1, student_feature.shape[1]), device=self.device) for class_feature in class_features]


                ''' create positive pairs and negative pairs of each class '''
                features_pos, features_neg = [[] for _ in range(self.num_classes)], [[] for _ in range(self.num_classes)]
                for i in range(self.num_classes):
                    features_pos[i] = torch.mean(class_features[i], dim=0).view(1, -1)
                    features_neg[i] = torch.stack([torch.mean(class_features[j], dim=0) for j in range(self.num_classes) if j != i])


                # create the positive and negative pairs for each data
                pos_pairs, neg_pairs = [], []
                for k in range(self.batch_size):
                    pos_pairs.append(features_pos[y[k]])
                    neg_pairs.append(features_neg[y[k]])

                pos_pairs = torch.stack(pos_pairs)
                neg_pairs = torch.stack(neg_pairs)

                pos_sim = self.cosine_sim(student_feature.unsqueeze(1), pos_pairs)
                neg_sim = self.cosine_sim(student_feature.unsqueeze(1), neg_pairs)

                logits = torch.cat([pos_sim, neg_sim], dim=1).to(self.device)
                logits /= self.T

                target = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)
                loss_con = F.cross_entropy(logits, target)
 

                loss_cls = 0
                for idx, (sig, value) in enumerate(self.knowledge_pool.items()):
                    _teacher = value["model"].to(self.device)
                    _teacher.eval()

                    weight = self.label_weights[y][:, idx].reshape(-1, 1)
                    weight = torch.tensor(weight.squeeze(), dtype=torch.float32, device=self.device)

                    teacher_logit, _ = _teacher(gen_output)
                    loss_cls += torch.mean(F.cross_entropy(teacher_logit.detach(), y_input) * weight)
                    _teacher.to("cpu")                                                  # move model back to cpu to save memory

                loss = self.ensemble_gamma * loss_con + self.ensemble_beta * loss_cls + self.ensemble_eta * loss_div
                loss_con_total.append(loss_con.item() * self.ensemble_gamma)
                loss_cls_total.append(loss_cls.item() * self.ensemble_beta)
                loss_div_total.append(loss_div.item() * self.ensemble_eta)
                loss_G_total.append(loss.item())

                self.optimizer_cgan.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 10)               # clip the gradient to prevent exploding
                self.optimizer_cgan.step()

            # ''' save the images from the generator '''
            gen_output = F.interpolate(gen_output, scale_factor=2, mode='bilinear', align_corners=False)
            save_image(make_grid(gen_output[:8], nrow=8, normalize=True), f"./figures/gen_output.png")

            ''' Train student (global model) '''
            generator.eval()
            global_model.train()

            ''' Sample new data '''
            # y = np.random.choice(self.qualified_labels, self.gen_batch_size)
            y = self.generate_labels(self.gen_batch_size, total_label_weights)
            y_input = F.one_hot(torch.tensor(y), num_classes=self.args.num_classes).type(torch.float32).to(self.device)

            for _ in range(self.inner_round_d):
                z = torch.randn((self.gen_batch_size, self.z_dim), device=self.device)
                gen_output = generator(z, y_input)
                student_logit, _ = global_model(gen_output)

                t_logit_merge = 0
                with torch.no_grad():
                    for idx, (sig, value) in enumerate(self.knowledge_pool.items()):
                        _teacher = value["model"].to(self.device)
                        _teacher.eval()

                        weight = self.label_weights[y][:, idx].reshape(-1, 1)
                        expand_weight = np.tile(weight, (1, self.num_classes))

                        teacher_logit, _ = _teacher(gen_output)

                        ''' knowledge distillation loss '''
                        t_logit_merge += teacher_logit.detach() * torch.tensor(expand_weight, dtype=torch.float32, device=self.device)
                        _teacher.to("cpu")                                                  # move model back to cpu to save memory

                loss_KD = F.kl_div(F.log_softmax(student_logit, dim=1)/self.T, F.softmax(t_logit_merge, dim=1)/self.T, reduction='batchmean')
                loss_KD_total.append(loss_KD.item())

                self.optimizer_server.zero_grad()
                loss_KD.backward()
                torch.nn.utils.clip_grad_norm_(global_model.parameters(), 10)               # clip the gradient to prevent exploding
                self.optimizer_server.step()

            val_acc = self.evaluate(global_model, self.valLoader, payloads)
            if val_acc > state["best_val_acc"]:
                state["best_val_acc"] = val_acc
                state["best_server_model"] = copy.deepcopy(global_model.state_dict())
                state["best_generator"] = copy.deepcopy(generator.state_dict())

            pbar.set_postfix({
                "loss_KD": np.mean(loss_KD_total),
                "loss_G": np.mean(loss_G_total),
                "loss_con": np.mean(loss_con_total),
                "loss_cls": np.mean(loss_cls_total),
                "loss_div": np.mean(loss_div_total),
                "val_acc": val_acc
            })

        # self.scheduler_cgan.step()
        # self.scheduler_server.step()

        # restore the best model
        generator.load_state_dict(state["best_generator"])
        global_model.load_state_dict(state["best_server_model"])
        global_model.eval()
        print("[Server | E. Distillation] Best val_acc: {:.4f}".format(state["best_val_acc"]))

        test_acc = self.evaluate(global_model, self.testLoader, payloads)
        print("[Server | Generative Knowledge Distilaltion]  After test_acc: {:.4f}".format(test_acc))
        return np.mean(loss_G_total), np.mean(loss_con_total), np.mean(loss_cls_total), np.mean(loss_div_total), np.mean(loss_KD_total), state["best_val_acc"], global_model.state_dict().values()


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


    def get_label_weights_from_knowledge_pool(self, knowledge_pool):
        MIN_SAMPLES_PER_LABEL = 1
        label_weights = np.zeros((self.args.num_classes, len(knowledge_pool)))

        for i, (sig, value) in enumerate(knowledge_pool.items()):
            for label, num_data in value["data_summary"].items():
                label_weights[label, i] += num_data  

        qualified_labels = np.where(label_weights.sum(axis=1) >= MIN_SAMPLES_PER_LABEL)[0]
        for i in range(self.args.num_classes):
            if np.sum(label_weights[i], axis=0) > 0:                    # avoid division by zero (lack of data at current round)
                label_weights[i] /= np.sum(label_weights[i], axis=0)
            else:
                label_weights[i] = 0

        return label_weights, qualified_labels


    def combine_weights(self, weights_1, weights_2):
        ''' Combine two weight tensors '''
        weight = weights_1 + weights_2

        if weight.ndim > 1:
            for i in range(len(weight)):
                if np.sum(weight[i], axis=0) > 0:                    # avoid division by zero (lack of data at current round)
                    weight[i] /= np.sum(weight[i], axis=0)
                else:
                    weight[i] = 0
        else:
            weight /= np.sum(weight, axis=0)

        return weight


    def get_age_weight(self,):
        ''' Calculate the age weight for each model in the knowledge pool '''
        age_weights = np.zeros(len(self.knowledge_pool))
        for idx, (sig, value) in enumerate(self.knowledge_pool.items()):
            if value["online_age"] > 0:
                age_weights[idx] = math.exp(value["online_age"] * self.online_rate)
            else:
                age_weights[idx] = math.exp(value["offline_age"] * self.offline_rate)

        ''' Normalize the age weight '''
        age_weights /= np.sum(age_weights, axis=0)
        return age_weights


    def get_importance_weight(self,):
        ''' Importance weight for each model can be approximated by the sum of the label weights '''
        label_weights, _ = self.get_label_weights_from_knowledge_pool(self.knowledge_pool)
        importance_weights = np.sum(label_weights, axis=0)

        ''' Normalize the importance weight '''
        importance_weights /= np.sum(importance_weights, axis=0)
        return importance_weights


    def update_knowledge_pool(self, client_list, rounds, global_payload=None):
        for client in client_list:
            status = client.status
            acc = client.test(payload=global_payload)["test_acc"]
            # print("client {} | status: {} | performance: {:.4f}".format(client.cid, status, acc))

            signature = "{cid}-{data}".format(cid=client.cid, data=str(list(client.data_distribution.keys())))
            # print(client.data_distribution)

            if signature in self.knowledge_pool:                                    # update the snapshot in knowledge pool
                item = self.knowledge_pool[signature]
                if status == "online":
                    item["model"] = copy.deepcopy(client.model)                     # move model to cpu to save memory
                    item["payload"] = copy.deepcopy(client.local_payload)                 # local objects (e.g., logits, protos, etc.)
                    item["data_summary"] = client.data_distribution
                    item["num_data"] = len(client.trainLoader.dataset)
                    item["performance"] = acc
                    item["online_age"] += 1
                    item["offline_age"] = 0
                else:
                    item["online_age"] = 0
                    item["offline_age"] += 1
            else:
                if status == "online":                                              # new to the knowledge pool
                    self.knowledge_pool[signature] = {
                        "model": copy.deepcopy(client.model),                       # move model to cpu to save memory
                        "payload": copy.deepcopy(client.local_payload),                   # local objects (e.g., logits, protos, etc.)
                        "data_summary": client.data_distribution,
                        "num_data": len(client.trainLoader.dataset),
                        "performance": acc,
                        "online_age": 1,
                        "offline_age": 0,
                    }


    def show_knowledge_pool(self,):
        print("\n-----> Knowledge Pool Status, Size: {}".format(len(self.knowledge_pool)))
        for key, value in self.knowledge_pool.items():
            line_head = Fore.YELLOW + "[Tag: {}]".format(key)
            line_mid1 = "Acc.: {:.4f}".format(value["performance"]) + Fore.RESET
            line_mid2 = "{}On. Age: {}".format(Fore.GREEN if value["online_age"] > 0 else Fore.LIGHTBLACK_EX, value["online_age"]) + Fore.RESET
            line_tail = "{}Off. Age: {}".format(Fore.RED if value["offline_age"] > 0 else Fore.LIGHTBLACK_EX, value["offline_age"]) + Fore.RESET
            print("{:<55} {:<25} {:<26} {:<27}".format(line_head, line_mid1, line_mid2, line_tail))
        print("-" * 101 + "\n")


    def aggregate_from_knowledge_pool(self, rounds) -> np.ndarray:
        ''' Aggregate with all models in the knowledge pool (no selection) '''
        client_numData_model_pair = [
            (value["model"].to(self.device), value["num_data"]) for sig, value in self.knowledge_pool.items()
        ]

        # Create a list of weights, each multiplied by the related number of examples
        model_weights = [
            [model.state_dict()[params] for params in model.state_dict()] for model, _ in client_numData_model_pair
        ]

        # Calculate the totol number of examples used during training
        num_examples_total = sum([num_examples for _, num_examples in client_numData_model_pair])

        # Calculate the weights of each model
        dataset_weights = [
            num_examples / num_examples_total for _, num_examples in client_numData_model_pair
        ]

        # Get the age weight of each model
        age_weights = self.get_age_weight()

        # Get the importance weight of each model
        importance_weights = self.get_importance_weight()

        # Calculate the final weights
        final_weights = self.age_ld * (torch.tensor(age_weights, device=self.device) * torch.tensor(dataset_weights, device=self.device)) + \
                         + self.impt_ld * torch.tensor(importance_weights, device=self.device)
        final_weights /= torch.sum(final_weights)

        # Compute average weight of each layer using the findal weights
        weights_prime = [
            reduce(torch.add, [w * weight for w, weight in zip(layer_updates, final_weights)])
            for layer_updates in zip(*model_weights)
        ]

        # Aggregate the payloads (class-wise by final weights, logits, protos, etc.)
        '''
        client_1_payload = {
            "data": {
                "class_0": [0.1, 0.2, 0.3],
                "class_1": [0.4, 0.5, 0}
        
        '''
        payloads_prime = {}
        if next(iter(self.knowledge_pool.items()))[1]["payload"]["type"] != "None":
            payloads_prime = self.fl_strategy.payload_aggregation(
                local_payload_list={i: value["payload"]["data"] for i, (sig, value) in enumerate(self.knowledge_pool.items())},
                client_weight_list = {i: final_weights[i] for i in range(len(self.knowledge_pool))}
                # client_weight_list={i: {j: payload_weights[j][i] for j in range(self.args.num_classes)} for i in range(len(self.knowledge_pool))}  # Reshape to match the number of classes
            )

        return weights_prime, payloads_prime
    

    def evaluate(self, server_model, testLoader, payloads=None) -> Tuple[float, float]:
        avg_acc = []
        for name, loader in testLoader.items():
            acc_indi = self.fl_strategy._test(server_model, loader, payloads)["test_acc"]
            avg_acc.append(acc_indi)
        return np.mean(avg_acc)