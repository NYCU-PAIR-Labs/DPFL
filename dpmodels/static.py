import random
import numpy as np
from dpmodels.dpmodel import BaseDPModel



class static(BaseDPModel):
    def __init__(self, args):
        super().__init__(args)

        self.num_clients = args.num_clients
        self.num_rounds = args.num_rounds

        self.client_state = np.zeros((self.num_clients, self.num_rounds))
        self.client_state = self.set_pattern()

    
    ''' Schedule clients for each round '''
    def set_pattern(self,):
        random.seed(self.args.seed+2)
        np.random.seed(self.args.seed+2)

        self.client_state[:, :] = 1
        return self.client_state