import random
import numpy as np
from dpmodels.dpmodel import BaseDPModel


class incremental_arrival(BaseDPModel):
    def __init__(self, args):
        super().__init__(args)

        self.num_clients = args.num_clients
        self.num_rounds = args.num_rounds
        self.round_start = args.round_start
        self.initial_clients = args.initial_clients
        self.interval = args.interval
        self.clients_per_interval = args.clients_per_interval

        self.client_state = np.zeros((self.num_clients, self.num_rounds))
        self.client_state = self.set_pattern()

    
    ''' Schedule clients for each round '''
    def set_pattern(self,):
        random.seed(self.args.seed+2)
        np.random.seed(self.args.seed+2)

        ''' There are `inital_clients` in the current environment. 
            The rest of the clients will arrive incrementally from the `round_start` round. 
            
            For example, if there are 3 clients and 6 rounds, the client_state matrix will be:
            ------------------------------------
            |  Client  |         Round         |
            ------------------------------------
            | client 0 | 1 | 1 | 1 | 1 | 1 | 1 |
            | client 1 | 0 | 0 | 1 | 1 | 1 | 1 |
            | client 2 | 0 | 0 | 0 | 0 | 1 | 1 |
        '''

        self.client_id = np.arange(self.num_clients)

        ''' The `initial_clients` clients will participate in all rounds. '''
        for c in range(self.initial_clients):
            self.client_state[self.client_id[c], :] = 1

        ''' The rest of the clients will arrive incrementally from the `round_start` round.'''
        i = self.round_start
        client_idx = self.initial_clients

        while i < self.num_rounds:
            for c in range(self.clients_per_interval):
                if client_idx < self.num_clients:
                    self.client_state[self.client_id[client_idx], i:] = 1
                    client_idx += 1
            i += self.interval

        return self.client_state