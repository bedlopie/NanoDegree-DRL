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


class Actor_flex(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layer=[], activation_fn=F.relu, output_fn=torch.tanh):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layer (array[int]): Number of nodes in hidden layers

        """
        super(Actor_flex, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.activation_fn = activation_fn
        self.output_fn = output_fn

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
        state = self.activation_fn(self.layers[0](state))
        for layer in self.layers[1:-1]:
            state = self.activation_fn(layer(state))
        
        return self.output_fn(self.layers[-1](state))


class Critic_flex_maddpg(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layer=[], activation_fn=F.relu):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """

        super(Critic_flex_maddpg, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.activation_fn = activation_fn

        if hidden_layer == []:
            raise ("Hidden Layer of Critic Network cannot be empty, at least one hiiden layer is needed to incorporate actions")
        
        layer_list_in = [state_size] + [hidden_layer[0]+action_size] + hidden_layer[1:]
        layer_list_out = hidden_layer + [1]
        # self.bn = nn.BatchNorm1d(layer_list_out[0])
        self.layers = nn.ModuleList([self.reset_parameters(nn.Linear(layer_in, layer_out)) for layer_in, layer_out in zip(layer_list_in, layer_list_out)])
        self.reset_parameters(self.layers[-1], fixed=(-3e-3, 3e-3))

    def reset_parameters(self, layer, fixed=None):
        if fixed == None :
            fixed = hidden_init(layer)     
        layer.weight.data.uniform_(*fixed)
        return layer

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = self.activation_fn(self.layers[0](state))
        state = torch.cat((xs, action), dim=1)
        for layer in self.layers[1:-1]:
            state = self.activation_fn(layer(state))
        
        return self.layers[-1](state)
        