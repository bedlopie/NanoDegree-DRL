import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import FC_Network
from MCTS import MCTS

import torch
import torch.nn.functional as F
import torch.optim as optim


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed, GPU, Value_network, LR_VALUE, Action_network, LR_ACTION):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """

        def torch_device(GPU_flag):
            if GPU_flag == True:
                if torch.cuda.is_available():
                    print("Running on GPU")
                    return torch.device("cuda:0")

            print("Running on CPU")
            return torch.device("cpu")

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random_seed
        random.seed(random_seed)
        self.device = torch_device(GPU)

        self.root = MCTS()

        # Definition of the Value Network
        self.NN_Value = FC_Network(state_size, 1, random_seed, hidden_layer=Value_network).to(self.device)
        self.NN_Value_optimizer = optim.Adam(self.NN_Value.parameters(), lr=LR_VALUE)

        # Definition of the Action Network
        self.NN_Action = FC_Network(state_size, action_size, random_seed, hidden_layer=Action_network).to(self.device)
        self.NN_Action_optimizer = optim.Adam(self.NN_Action.parameters(), lr=LR_ACTION)



