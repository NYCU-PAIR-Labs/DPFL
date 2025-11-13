from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np

'''
    This is the implementation of the client scheduler. There are 6 types of client dynamics:
    - static: all clients participate in all rounds
    - incremental_arrival: clients arrive incrementally from a specific round
    - incremental_departure: clients depart incrementally from a specific round
    - round-robin: clients participate in a round-robin fashion
    - random: clients participate randomly in each round        
    - markov: clients participate based on a Markov chain

    - client_state: a matrix to store the state of clients in each round
    ------------------------------------
    |  Client  |         Round         |
    ------------------------------------
    | client 0 | 1 | 1 | 0 | 0 | 0 | 0 |
    | client 1 | 0 | 0 | 1 | 1 | 0 | 0 |
    | client 2 | 0 | 0 | 0 | 0 | 1 | 1 |
'''


class BaseDPModel(ABC):
    """
    Abstract base class for client participation scheduling in federated learning.
    Subclass this and implement `set_pattern()` to define your participation logic.

    Attributes:
        args: Namespace or dict containing configuration (must include num_clients, num_rounds, etc.)
        client_state: np.ndarray of shape (num_clients, num_rounds), 1 if client participates in round, else 0

    Example:
        class MyPattern(BaseDPModel):
            def set_pattern(self):
                # Custom logic to fill self.client_state
                ...
                return self.client_state
    """

    def __init__(self, args):
        self.num_clients = args.num_clients
        self.num_rounds = args.num_rounds
        self.args = args
        self.client_state = np.zeros((self.num_clients, self.num_rounds), dtype=int)        # build a matrix to store the state of clients in each round


    ''' Schedule clients for each round '''
    @abstractmethod
    def set_pattern(self):
        """
        Define the participation pattern by filling self.client_state.
        Must return the participation matrix (np.ndarray).
        """

        raise NotImplementedError("set_pattern() must be implemented in your strategy.")
        

    ''' Update the clients for the current round '''
    def update(self, round):
        """
        Get active and inactive client indices for a given round (1-based).
        Returns:
            active_ids: np.ndarray of active client indices
            inactive_ids: np.ndarray of inactive client indices
        """
        current_state = self.client_state[:, round-1]

        active_ids = np.where(current_state == 1)[0]
        inactives_ids = np.where(current_state == 0)[0]
        return active_ids, inactives_ids

    
    def visualize(self):
        """
        Optional: Visualize the participation matrix.
        """
        plt.imshow(self.client_state, aspect='auto', cmap='Greys')
        plt.xlabel("Round")
        plt.ylabel("Client")
        plt.title("Client Participation Matrix")
        plt.show()


# scheduler = ClientScheduler(args)
# print(scheduler.schedule())