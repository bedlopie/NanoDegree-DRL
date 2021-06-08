import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class FC_Network(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layer=[]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layer (array[int]): Number of nodes in hidden layers

        """
        super(FC_Network, self).__init__()
        self.seed = torch.manual_seed(seed)

        layer_list = [state_size] + hidden_layer + [action_size]
        # self.bn = nn.BatchNorm1d(layer_list[1])
        self.layers = nn.ModuleList([self.reset_parameters(nn.Linear(layer_in, layer_out)) for layer_in, layer_out in zip(layer_list[:-1], layer_list[1:])])
        self.reset_parameters(self.layers[-1], fixed=(-3e-3, 3e-3))

    def reset_parameters(self, layer, fixed=None):
        if fixed == None :
            fixed = hidden_init(layer)     
        layer.weight.data.uniform_(*fixed)
        return layer

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        state = F.relu(self.layers[0](state))
        for layer in self.layers[1:-1]:
            state = F.relu(layer(state))
        
        return torch.tanh(self.layers[-1](state))

