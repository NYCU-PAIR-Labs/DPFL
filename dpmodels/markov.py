import random
import numpy as np
from dpmodels.dpmodel import BaseDPModel



class markov(BaseDPModel):
    def __init__(self, args):
        super().__init__(args)

        self.num_clients = args.num_clients
        self.num_rounds = args.num_rounds
        # self.round_start = args.round_start

        self.client_state = np.zeros((self.num_clients, self.num_rounds))
        self.client_state = self.set_pattern()

    
    ''' Schedule clients for each round '''
    def set_pattern(self,):
        random.seed(self.args.seed+2)
        np.random.seed(self.args.seed+2)


        ''' Clients participate based on a Markov chain.
            There are 2 states: active and inactive.
            - Active: `active_prob` probability to stay active in the next round, `1-active_prob` probability to become inactive.
            - Inactive: `inactive_prob` probability to stay inactive in the next round, `1-inactive_prob` probability to become active.

                (1 - active_prob)
                    --------
                    |      |
                    v      |  
            ------------  ==> active_prob ===>  ----------
            | inactive |                        | active |
            ------------  <== inactive_prob ==  ----------
                                                |      ^
                                                |      |
                                                --------
                                        (1 - inactive_prob)

        '''

        transition_matrix = np.array([
            [0.2, 0.8],                     # inactive -> active, inactive -> inactive
            [0.8, 0.2],                     #   active -> active,   active -> inactive
        ])

        ''' Randomly assign the initial state (round=0) for each client '''
        self.client_state[:, 0] = np.random.choice([0, 1], self.num_clients)

        ''' Update the state of clients based on the transition matrix '''
        for r in range(1, self.num_rounds):
            for c in range(self.num_clients):
                self.client_state[c, r] = np.random.choice([1, 0], p=transition_matrix[int(self.client_state[c, r-1])])

        # check if there is at least one client participating in each round
        for r in range(self.num_rounds):
            if np.sum(self.client_state[:, r]) == 0:
                c = np.random.randint(0, self.num_clients)
                self.client_state[c, r] = 1

        return self.client_state