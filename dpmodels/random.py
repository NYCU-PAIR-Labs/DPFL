import random as rd
import numpy as np
from dpmodels.dpmodel import BaseDPModel



class random(BaseDPModel):
    def __init__(self, args):
        super().__init__(args)

        self.num_clients = args.num_clients
        self.num_rounds = args.num_rounds
        # self.round_start = args.round_start

        self.client_state = np.zeros((self.num_clients, self.num_rounds))
        self.client_state = self.set_pattern()

    
    ''' Schedule clients for each round '''
    def set_pattern(self,):
        rd.seed(self.args.seed+2)
        np.random.seed(self.args.seed+2)

        ''' Randomly assign clients to participate in each round. 
            For example, if there are 3 clients and 6 rounds, the client_state matrix will be:
            ------------------------------------
            |  Client  |         Round         |
            ------------------------------------
            | client 0 | 1 | 0 | 1 | 0 | 0 | 1 |
            | client 1 | 0 | 1 | 0 | 1 | 1 | 0 |
            | client 2 | 0 | 0 | 0 | 0 | 1 | 1 |
        '''

        # self.client_state[:, :self.round_start] = 1

        for c in range(self.num_clients):
            self.client_state[c, :] = np.random.randint(0, 2, self.num_rounds)

        # check if there is at least one client participating in each round
        for r in range(self.num_rounds):
            if np.sum(self.client_state[:, r]) == 0:
                c = np.random.randint(0, self.num_clients)
                self.client_state[c, r] = 1

        return self.client_state