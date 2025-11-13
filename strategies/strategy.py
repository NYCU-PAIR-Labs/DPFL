from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch

class Strategy(ABC):
    """
    Abstract base class for federated learning strategies.
    To implement a new FL algorithm, subclass this and implement all abstract methods.
    """

    def __init__(self, device: torch.device, args: Any) -> None:
        """
        Args:
            global_payload: Optional dictionary to hold local auxiliary objects (e.g., logits, prototypes).
            local_payload: Optional dictionary to hold local payload data.
            device: torch.device for computation.
            args: Namespace or dict of experiment arguments.
        """

        self.global_payload: Optional[Dict[str, Any]] = {}
        self.local_payload:  Optional[Dict[str, Any]] = {
            "type": "None",     # Type of payload (e.g., logits, prototypes)
            "data": None,       # Actual payload data
        }
        self.device = device
        self.args = args

    @abstractmethod
    def _initialization(self, **kwargs) -> None:
        """
        Called once before training starts.
        """
        raise NotImplementedError("_initialization() must be implemented in your strategy.")

    @abstractmethod
    def _server_train_func(self, cid: int, rounds: int, client_list: List[Any], **kwargs) -> None:
        """
        Defines the training workflow for the clients at the server.
        """
        raise NotImplementedError("_server_train_func() must be implemented in your strategy.")

    @abstractmethod
    def _server_agg_func(self, rounds: int, client_list: List[Any], active_clients: List[Any], global_model: torch.nn.Module) -> None:
        """
        Defines the aggregation workflow at the server.
        return:
            new_weights: The new weights for the global model. [model.state_dict()]
        """
        raise NotImplementedError("_server_agg_func() must be implemented in your strategy.")

    @abstractmethod
    def _aggregation(
        self,
        server_round: int,
        client_models: List[Tuple[np.ndarray, int]],
    ) -> np.ndarray:
        """ Defines the actual aggregation process.
        
        Parameters
        ----------
        server_round: int
            The current round of federated learning.
        client_models:
            A list of client models and their respective amount of data
        
        return:
            new_weights: The new weights for the global model. [model.state_dict()]
        """
        raise NotImplementedError("_aggregation() must be implemented in your strategy.")

    @abstractmethod
    def _get_local_payload(self) -> Dict[str, Any]:
        """
        Get the current payload from the strategy.
        """
        raise NotImplementedError("_get_payload() must be implemented in your strategy.")

    @abstractmethod
    def _train(self, model: torch.nn.Module, trainLoader, optimizer, num_epochs, **kwargs) -> Dict[str, float]:
        """
        Defines the actual training process for the clients.
        return:
            dict: A dictionary containing the training loss and accuracy.
                - {"train_loss": float, "train_acc": float}
        """
        raise NotImplementedError("_train() must be implemented in your strategy.")
    
    @abstractmethod
    def _test(self, model: torch.nn.Module, testLoader, **kwargs) -> Dict[str, float]:
        """
        Defines the actual testing process for the clients.
        return:
            dict: A dictionary containing the testing loss and accuracy.
                - {"test_loss": float, "test_acc": float}
        """
        raise NotImplementedError("_test() must be implemented in your strategy.")