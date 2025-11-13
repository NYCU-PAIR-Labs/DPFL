from typing import List, Tuple, Dict, Any
from .strategy import Strategy
import numpy as np
import torch
from functools import reduce
import torch.nn.functional as F

class FedAvg(Strategy):
    def __init__(self, device, args, params):
        super().__init__(device=device, args=args)


    def _initialization(self, **kwargs) -> None:
        pass


    def _server_train_func(self, cid, rounds, client_list, **kwargs) -> None:
        ''' Train function for the server to orchestrate the training process '''
        result = client_list[cid].train()
        train_loss, train_acc = result["train_loss"], result["train_acc"]
        print("[Client {}] loss: {:.4f}, acc: {:.4f}".format(cid, train_loss, train_acc))


    def _server_agg_func(self, rounds, client_list, active_clients, global_model):
        ''' Aggregate function for the server '''
        new_weights = self._aggregation(client_list)
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

        return {
            "train_loss": np.mean(total_loss),
            "train_acc": np.mean(total_correct)
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
        return {
            "test_loss": np.mean(total_loss),
            "test_acc": np.mean(total_correct),
            "class_acc": class_acc
        }


    def _aggregation(self, client_list, average_mode="weighted_sum", mode="active") -> np.ndarray:
        ''' Aggregate all clients with status "online" '''
        client_numData_model_pair = []
        if mode == "active":
            if average_mode == "weighted_sum":
                client_numData_model_pair = [
                    (c.model, len(c.trainLoader.dataset)) 
                    for c in client_list if c.status == "online"
                ]
            elif average_mode == "uniform":
                client_numData_model_pair = [
                    (c.model, 1) 
                    for c in client_list if c.status == "online"
                ]
            else:
                raise ValueError("Invalid averaging mode")

        elif mode == "all":
            client_numData_model_pair = [
                (c.model.to(self.device), len(c.trainLoader.dataset)) 
                for c in client_list
            ]
        else:
            raise ValueError("Invalid aggregation mode")
        
        if len(client_numData_model_pair) == 0:
            return None

        # Calculate the totol number of examples used during training
        num_examples_total = sum([num_examples for _, num_examples in client_numData_model_pair])

        # Create a list of weights, each multiplied by the related number of examples
        weighted_weights = [
            [model.state_dict()[params] * num_examples for params in model.state_dict()] for model, num_examples in client_numData_model_pair
        ]

        # Compute average weight of each layer
        weights_prime = [
            reduce(torch.add, layer_updates) / num_examples_total
            for layer_updates in zip(*weighted_weights)
        ]
        return weights_prime
    

    def _get_local_payload(self) -> Dict[str, Any]:
        """
        Get the current payload from the strategy.
        """
        return self.local_payload