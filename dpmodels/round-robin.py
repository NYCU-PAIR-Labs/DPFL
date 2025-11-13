import random
import numpy as np
from dpmodels.dpmodel import BaseDPModel



class round_robin(BaseDPModel):
    def __init__(self, args):
        super().__init__(args)

        self.num_clients = args.num_clients
        self.num_rounds = args.num_rounds
        # self.round_start = args.round_start
        self.interval = args.interval
        self.clients_per_interval = args.clients_per_interval
        self.overlap_clients = args.overlap_clients

        self.client_state = np.zeros((self.num_clients, self.num_rounds))
        self.client_state = self.set_pattern()

    
    ''' Schedule clients for each round '''
    def set_pattern(self,):
        random.seed(self.args.seed+2)
        np.random.seed(self.args.seed+2)

        ''' There are `num_clients` clients need to be iterated participating with `interval` rounds.
            The overlap_clients parameter controls how many clients overlap between consecutive intervals.
            For example, if overlap_clients=1, clients_per_interval=2, and interval=2, the client_state matrix will be:
            ------------------------------------
            |  Client  |         Round         |
            ------------------------------------
            | client 0 | 1 | 1 | 0 | 0 | 0 | 0 |
            | client 1 | 1 | 1 | 1 | 1 | 0 | 0 |
            | client 2 | 0 | 0 | 1 | 1 | 1 | 1 |
            | client 3 | 0 | 0 | 0 | 0 | 1 | 1 |
        '''

        i = 0                   # self.round_start
        start_client = 0

        while i < self.num_rounds:
            # Schedule clients_per_interval clients for this interval
            for c in range(self.clients_per_interval):
                client_idx = (start_client + c) % self.num_clients
                self.client_state[client_idx, i: min(i+self.interval, self.num_rounds)] = 1

            # Move to next interval, accounting for overlap
            # If overlap_clients > 0, the next interval starts overlap_clients back
            start_client += (self.clients_per_interval - self.overlap_clients)
            i += self.interval
            if i >= self.num_rounds:
                break

        return self.client_state