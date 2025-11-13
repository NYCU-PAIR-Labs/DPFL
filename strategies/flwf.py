from typing import List, Tuple, Dict, Any
from .fedavg import FedAvg
import numpy as np
import copy
import torch
from functools import reduce
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer

'''
Hyperparameters (From paper):
    - temperature: 2
'''

class FLwF(FedAvg):
    def __init__(self, device, args, params):
        super().__init__(device=device, args=args, params=params)
        self.prev_global_model = None
        self.alpha = params["alpha"]                # 0.1
        self.T = params["T"]                        # temperature-scaled logit for distillation loss
        

    def _initialization(self, **kwargs) -> None:
        pass


    def _server_train_func(self, cid, rounds, client_list, **kwargs) -> None:
        ''' Train function for the server to orchestrate the training process '''
        # train_loss, train_acc, dist_loss = self.client_list[i].train(prev_global_model=self.strategy.prev_global_model, task_id=self._current_tid)
        global_model = kwargs["global_model"]

        result = client_list[cid].train(prev_global_model=global_model, task_id=rounds)
        train_loss, train_acc, dist_loss = result["train_loss"], result["train_acc"], result["dist_loss"]
        print("[Client {}] loss: {:.4f}, dist_loss: {:.4f}, acc: {:.4f}".format(cid, train_loss, dist_loss, train_acc))


    def _server_agg_func(self, rounds, client_list, active_clients, global_model):
        ''' Aggregate function for the server '''
        new_weights = self._aggregation(client_list)
        return new_weights


    def _train(self, model, trainLoader, optimizer, num_epochs, **kwargs) -> Tuple[float, float]:
        prev_global_model, task_id = kwargs["prev_global_model"], kwargs["task_id"]

        ''' Train function for the client '''
        prev_global = copy.deepcopy(prev_global_model).to(self.device)
        prev_global.eval()

        model.train()
        for i in range(num_epochs):
            total_loss, total_correct = [], []
            dist_loss = []
            for x, label in trainLoader:
                x, label = x.to(self.device), label.to(self.device)
                logit_student, _ = model(x)

                predict = torch.argmax(logit_student.data, 1)
                total_correct.append((predict == label).sum().item() / len(predict))

                ce_loss = F.cross_entropy(logit_student, label)

                ''' Distillation loss with global model '''
                if task_id > 1:
                    with torch.no_grad():
                        logit_teacher, _ = prev_global(x)

                    distillation_loss = self.distillation_loss(logit_student, logit_teacher)
                    dist_loss.append(distillation_loss.item())
                else:
                    distillation_loss = 0
                    dist_loss = [0]

                loss = ce_loss + self.alpha * distillation_loss
                total_loss.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)               # clip the gradient to prevent exploding
                optimizer.step()

        # return np.mean(total_loss), np.mean(total_correct), np.mean(dist_loss)
        return {
            "train_loss": np.mean(total_loss),
            "train_acc": np.mean(total_correct),
            "dist_loss": np.mean(dist_loss)
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


    def distillation_loss(self, student_logits, teacher_logits):
        teacher_logits = F.softmax(teacher_logits / self.T, dim=1)
        student_logits = F.log_softmax(student_logits / self.T, dim=1)
        return -torch.mean(torch.sum(student_logits * teacher_logits, dim=1, keepdim=False), dim=0, keepdim=False)
    

    def _get_local_payload(self) -> Dict[str, Any]:
        """
        Get the current payload from the strategy.
        """
        return self.local_payload