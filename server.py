from typing import Dict, List, Tuple
from colorama import Fore, Style
import numpy as np
import nni
import wandb
import copy
import torch
from torch.utils.data import ConcatDataset
import torch.nn.functional as F
from utils.base_client import BaseClient
from client import FLclient
from strategies.strategy import Strategy
from strategies.feddpfl import FedDPFL
from strategies.mifa import MIFA
from utils.data_model import DataModel
from utils.util import *



''' FL Server class '''
class FLserver(BaseClient):
    def __init__(self, clients: List[FLclient], testset: Dict[str, Dataset], strategy: Strategy, device: torch.device, params: Dict, args: Dict, sim=None):
        super().__init__(device, args)
        '''
        clients: a list of current clients
        dataLoader: dataloader for testing data
        dataLoader_indi: list of dataloader for individual classes [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        device: cpu or gpu
        args: arguments
        '''
        self.args = args
        self.client_list = clients
        self.active_clients = []
        self.inactive_clients = []

        self.completed_clients = set()
        self.delayed_clients = set()
        self._current_tid = 0
        self.testset = self.init_testset(testset)
        self.testLoader = None

        self.strategy = copy.deepcopy(strategy)
        self._strategy = strategy.__class__.__name__

        self.params = params
        self.device = device
        self.sim = sim
        self.payload = None
        

    def init_testset(self, testset):
        if self.args.incremental_type == "class-incremental":
            tmp_dataset = {}
            for name, dataset in testset.items():
                tmp_dataset[name] = DataModel().divide_dataset_to_incremental_partition(dataset, self.args)

            task_dataset = {
                f"task_{i}": {
                    name: tmp_dataset[name][f"task_{i}"] for name in testset.keys()
                }
                for i in range(1, 6)
            }
            return task_dataset
        else:
            return testset


    def get_current_dataLoader(self,):
        if self.args.incremental_type == "class-incremental":
            testset = {
                name: ConcatDataset([self.testset[f"task_{i}"][name] for i in range(1, self._current_tid + 1)])
                for name in self.testset["task_1"].keys()
            }
        else:
            testset = self.testset
    
        testLoader = {
            name: DataLoader(testset, batch_size=self.args.batch_size, pin_memory=True)
            for name, testset in testset.items()
        }
        return testLoader


    def update_incremental_data(self, rounds):
        ''' Update the server's data and client's data if is under class-incremental learning '''
        self._current_tid += 1
        self.testLoader = self.get_current_dataLoader()
        self.strategy.prev_global_model = copy.deepcopy(self.model)

        for i in range(len(self.client_list)):
            self.client_list[i].update_incremental_data()

        if rounds > 1:
            if self._strategy == "Target":
                self.strategy.data_generation(global_model=self.model, rounds=self._current_tid - 1)
                self.strategy.syn_data_loader = self.strategy.get_syn_data_loader(rounds=self._current_tid - 1)
                self.strategy.syn_iter = iter(self.strategy.syn_data_loader)


    def add_client(self, client: FLclient):
        ''' Add a client to the server '''
        self.client_list.append(client)
        self.set_client_model(client.cid, self.get_parameters())
        self.active_clients.append(client.cid)
        print("[+] Add client, cid:{}".format(client.cid))
        # print("Total clients: {}, {}".format(len(self.client_list), [c.cid for c in self.client_list]))


    def set_client_model(self, cid, parameters):
        ''' Set the model of the client '''
        for i in range(len(self.client_list)):
            if self.client_list[i].cid == cid:
                self.client_list[i].set_parameters(parameters)
                break


    ''' -------------------------------------------------- Core Functions -------------------------------------------------- '''

    def initialization(self,):
        self.active_clients = np.sort(self.active_clients)

        ''' Initialize the server's and client's local dataset '''
        self.update_incremental_data(rounds=0)

        ''' Initialization according to the strategy '''
        self.strategy._initialization(client_list=self.client_list, active_clients=self.active_clients, global_model=self.model)
        
        ''' Collect client's latency for deadline setting '''
        if self.sim:
            self.client_deadlines = {}
            self.aggregation_deadline = self.set_deadline([c.cid for c in self.client_list])

        ''' If DPFL, initialize the knowledge pool '''
        if self.args.kpfl:
            # self.mifa = MIFA(self.model, self.device, self.args)
            self.knowledge_pool = FedDPFL(self.strategy, self.device, self.args, self.params)
            self.knowledge_pool.initialization(global_model=self.model, client_list=self.client_list, testLoader=self.testLoader)

        print("\n" + Fore.RED + "[Server] `{}` initialization done".format(self._strategy) + Fore.RESET)


    def train_clients(self, rounds: int):
        ''' Train all clients with status "online" '''
        print(Fore.CYAN + "[Server] Start training client" + Fore.RESET)
        if self.sim:
            process_handles = {}                                                    # Store process references

        for i in self.active_clients:
            self.set_client_model(cid=i, parameters=self.get_parameters())          # set client model's weight
            if self.sim:
                # Schedule process with completion tracking
                self.client_list[i].reset_latency(),                                # Reset latency at the start of the process
                
                proc = self.sim.process(lambda sim=self.sim, cid=i, r=rounds, comp_list=self.completed_clients: (
                    self.strategy._server_train_func(cid=cid, rounds=r, client_list=self.client_list, global_model=self.model, global_payload=self.strategy.global_payload),
                    comp_list.add(cid),                             # Append AFTER training completes
                    # print(Fore.GREEN + f"[Sim] Client {cid} completed training at {format_sim_time(sim.now)}" + Fore.RESET)
                ))
                process_handles[i] = proc
            else:
                self.strategy._server_train_func(cid=i, rounds=rounds, client_list=self.client_list, global_model=self.model, global_payload=self.strategy.global_payload)

        ''' Ensure all scheduled training finishes in this virtual round '''
        if self.sim:
            aggregation_deadline = self.sim.now + self.aggregation_deadline
            print(Fore.YELLOW + f"Current: {format_sim_time(self.sim.now)}, Aggregation deadline (s): {format_sim_time(aggregation_deadline)}" + Fore.RESET)

            self.sim.run(until=aggregation_deadline)

            ''' # Mark client as online upon completion '''
            for cid in self.completed_clients:
                self.client_list[cid].status = "online"

            # Identify stragglers
            self.delayed_clients.update([cid for cid in self.active_clients if cid not in self.completed_clients])
            self.delayed_clients.difference_update(self.completed_clients)

            for cid in self.delayed_clients:
                self.client_list[cid].status = "missed"

            print(f"{Fore.BLACK}{format_sim_time(self.sim.now)} |{Fore.YELLOW} Completed clients: `{len(self.completed_clients)}`, {list(self.completed_clients)}{Fore.RESET}")
            print(f"{Fore.BLACK}{format_sim_time(self.sim.now)} |{Fore.YELLOW} Delayed clients: `{len(self.delayed_clients)}`, {list(self.delayed_clients)}{Fore.RESET}")
            self.completed_clients.clear()
            self.show_clients()

        ''' MIFA update '''
        # self.mifa.update_mifa(client=self.client_list[i], global_model=self.model)


    def aggregate_clients(self, rounds: int):
        ''' Aggregation function of different strategies '''
        new_weights = self.strategy._server_agg_func(rounds, self.client_list, self.active_clients, self.model)

        ''' DFPL: Aggregate from the knowledge pool '''
        if self.args.kpfl:
            ''' MIFA update '''
            # new_weights = self.mifa.aggregation_mifa(global_model=self.model, rounds=rounds)

            self.knowledge_pool.update_knowledge_pool(client_list=self.client_list, rounds=rounds, global_payload=self.strategy.global_payload)
            self.knowledge_pool.show_knowledge_pool()
            
            ''' aggregate from selecting models and local objects (e.g., logits, protos, etc.) in the knowledge pool '''
            agg_weights, agg_payloads = self.knowledge_pool.aggregate_from_knowledge_pool(rounds=rounds)
            self.strategy.global_payload = agg_payloads
            
            ''' set global model's weight for ensemble distillation '''
            self.set_parameters(agg_weights)
            
            loss_g, loss_con, loss_cls, loss_div, loss_kd, avg_acc, new_weights_prime = self.knowledge_pool.data_free_knowledge_distillation(
                global_model=self.model,
                payloads=self.strategy.global_payload,
                rounds=rounds,
            )
            print("[Server] loss_KD: {:.4f}, loss_G: {:.4f}, loss_con: {:.4f}, loss_cls: {:.4f}, loss_div: {:.4f}".format(loss_kd, loss_g, loss_con, loss_cls, loss_div))


            ''' rounds <= args.round_start '''
            if (self.args.dynamic_type in ["incremental-arrival", "incremental-departure"] and rounds <= 50) \
                or (self.args.dynamic_type in ["round-robin"] and rounds <= 10):
                print(Fore.CYAN + "[Server] Not updating the global model by DPFL" + Fore.RESET)
                pass
            else:
                new_weights = new_weights_prime

        if new_weights is None:
            print(Fore.CYAN + "[Server] No clients to aggregate from" + Fore.RESET)
            new_weights = self.get_parameters()                                 # Keep the current global model unchanged

        self.set_parameters(new_weights)                                        # set global model's weight
        print(Fore.CYAN + "[Server] Done aggregating client models" + Fore.RESET)


    def set_deadline(self, active_clients: List[int], mode="default"):
        ''' Collect client's latency for deadline setting '''
        if mode == "default":
            aggregation_deadline = 10.0

        elif mode == "by_percentile":
            for i in self.client_list:
                self.client_deadlines[i.cid] = i.dl_latency + i.up_latency + i.cp_latency

            client_deadlines = {cid: self.client_deadlines[cid] for cid in active_clients}
            sorted_deadlines = sorted(client_deadlines.values())
            index = round(len(sorted_deadlines) * self.args.deadline_percentile) - 1
            aggregation_deadline = sorted_deadlines[index] + self.sim.now + 0.001       # Add a small buffer to ensure all processes scheduled to finish at this time are included
            # print("Client deadlines (s):", self.client_deadlines)
        return aggregation_deadline


    ''' -------------------------------------------------- Core Functions -------------------------------------------------- '''


    def set_client_state(self, active_ids: List[int], inactive_ids: List[int]):
        ''' Ignore the delayed clients  '''
        active_ids = [cid for cid in active_ids if cid not in self.delayed_clients]
        inactive_ids = [cid for cid in inactive_ids if cid not in self.delayed_clients]
        self.set_active_client(active_ids)
        self.set_inactive_client(inactive_ids)


    def set_inactive_client(self, clients: List[int]):
        for i in range(len(self.client_list)):
            if self.client_list[i].cid in clients:
                self.client_list[i].status = "offline"

        self.inactive_clients = np.sort(clients)
        self.active_clients = np.sort([c for c in self.active_clients if c not in clients])
        print("[-] set inactive: {}".format(self.inactive_clients))


    def set_active_client(self, clients: List[int]):
        for i in range(len(self.client_list)):
            if self.client_list[i].cid in clients:
                self.client_list[i].status = "online"
                # if self._strategy != "FedProto":
                self.set_client_model(self.client_list[i].cid, self.get_parameters())   # 恢復訓練的 client 拿到最新的 model

        self.active_clients = np.sort(clients)
        self.inactive_clients = np.sort([c for c in self.inactive_clients if c not in clients])
        print("[+] set active: {}".format(self.active_clients))


    def show_clients(self,):
        # Determine the maximum width needed for client IDs
        # max_id_width = len(str(len(self.client_list) - 1)) + 1

        # print("Active: {}".format(self.active_clients))
        # print("Inactive clients: {}".format(self.inactive_clients))
        bar = "Client |"
        for i in range(len(self.client_list)):
            bar += f" {i} |"                            # Pad to max_id_width
        bar += "\n"
        line = "-" * len(bar) + "\n"
        content = "Status |"
        for i in range(len(self.client_list)):
            if self.client_list[i].status == "online":
                content += f" {Fore.LIGHTRED_EX}{'O':^{len(str(i))}}{Fore.RESET} |"
            elif self.client_list[i].status == "missed":
                content += f" {Fore.LIGHTYELLOW_EX}{'⚠︎':^{len(str(i))}}{Fore.RESET} |"
            else:
                content += f" {Fore.LIGHTBLACK_EX}{'X':^{len(str(i))}}{Fore.RESET} |"
        content += "\n"
        print(line + bar + line + content + line)


    def evaluate(self, rounds: int, model=None, tag="normal"):
        ''' Evaluate the global model on individual domains '''
        print("===== Evaluation =====")

        if tag == "normal":
            ''' Test performance on active models '''
            result = self.evaluate_each_domain(model)
        elif tag == "all":
            ''' Test performance on all models '''
            all_model = copy.deepcopy(self.model)
            all_model_params = self.strategy._aggregation(self.client_list, mode="all")
            set_param(model=all_model, parameters=all_model_params)
            result = self.evaluate_each_domain(all_model)
        else:
            raise ValueError("Non-supported tag")
            

        total_loss, total_acc = [], []
        total_class_acc = {i: [] for i in range(self.args.num_classes)}
        for name in self.testLoader.keys():
            loss_indi, acc_indi, class_indi = result[name]["loss"], result[name]["acc"], result[name]["class_acc"]
            total_loss.append(loss_indi)
            total_acc.append(acc_indi)
            for class_id in class_indi:
                total_class_acc[class_id].append(class_indi[class_id])
                wandb.log({
                    f"({name}) {class_id}_{tag}": class_indi[class_id],
                    "rounds": rounds,
                })
                
            print(Fore.GREEN + "[Server, {}] test_loss: {:.4f}, test_acc: {:.4f}".format(name, loss_indi, acc_indi) + Fore.RESET)
            wandb.log({
                f"({name})_loss_{tag}": loss_indi,
                f"({name})_acc_{tag}": acc_indi,
                "rounds": rounds,
            })
        
        ''' Calculate the average loss & accuracy '''
        total_loss = np.mean(total_loss)
        total_acc = np.mean(total_acc)
        print(Fore.GREEN + Style.BRIGHT + "[Server] Total_loss: {:.4f}, Total_acc: {:.4f}".format(total_loss, total_acc) + Style.RESET_ALL + Fore.RESET)
        wandb.log({
            f"total_loss_{tag}": total_loss,
            f"total_acc_{tag}": total_acc,
            "rounds": rounds,
        })

        ''' Report Accuracy for NNI '''
        if self.args.nni and tag == "normal":
            nni.report_intermediate_result(total_acc)
            if rounds == self.args.num_rounds:
                nni.report_final_result(total_acc)

        ''' Calculate each class's accuracy '''
        total_class_acc = {i: np.mean(total_class_acc[i]) for i in range(self.args.num_classes)}
        for class_id in total_class_acc:
            wandb.log({
            f"total_acc {class_id}_{tag}": total_class_acc[class_id],
            "rounds": rounds,
        })
        return result


    def evaluate_each_domain(self, model):
        ''' Evaluate the global model on individual domains
        Returns:
            result: {
                "MNIST": {
                    "loss": 0.1,
                    "acc": 0.9,
                    "class": {
                        0: 0.99,
                        1: 0.98,
                    }
                },
                "USPS": {
                    "loss": 0.2,
                    "acc": 0.8,
                },
            }
        '''
        result = {}
        for name, loader in self.testLoader.items():
            result_indi = self.test(model, loader)
            loss_indi, acc_indi, class_indi = result_indi["test_loss"], result_indi["test_acc"], result_indi["class_acc"]
            record = {
                "loss": loss_indi,
                "acc": acc_indi,
                "class_acc": class_indi,
            }
            result[name] = record
        return result
    

    def evaluate_each_client(self,):
        ''' Evaluate each client on testing data '''
        result = {
            name: {
                "loss": [],
                "acc": [],
                "class_acc": {
                    i: [] for i in range(self.args.num_classes)
                },
            } for name in self.testLoader.keys()
        }
    
        for i in tqdm(range(len(self.client_list)), leave=False):
            # if self.client_list[i].status == "online":
                client_result = self.evaluate_each_domain(self.client_list[i].model)
                for name in client_result:
                    result[name]["loss"].append(client_result[name]["loss"])
                    result[name]["acc"].append(client_result[name]["acc"])
                    for class_id in client_result[name]["class_acc"]:
                        result[name]["class_acc"][class_id].append(client_result[name]["class_acc"][class_id])

        for name in result:
            result[name]["loss"] = np.mean(result[name]["loss"])
            result[name]["acc"] = np.mean(result[name]["acc"])
            for class_id in result[name]["class_acc"]:
                result[name]["class_acc"][class_id] = np.mean(result[name]["class_acc"][class_id])
        return result


    def test(self, model,  dataLoader: DataLoader):
        return self.strategy._test(model, dataLoader, payload=self.strategy.global_payload)