import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor_flex(nn.Module):
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
        super(Actor_flex, self).__init__()
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


class Critic_flex(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layer=[]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic_flex, self).__init__()
        self.seed = torch.manual_seed(seed)
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
        xs = F.relu(self.layers[0](state))
        state = torch.cat((xs, action), dim=1)
        for layer in self.layers[1:-1]:
            state = F.relu(layer(state))
        
        return self.layers[-1](state)
        

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=600, fc2_units=450, fc3_units=450):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            fc3_units (int): Number of nodes in third hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        #self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        #self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        return torch.tanh(self.fc4(x))



class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=600, fc2_units=450, fc3_units=450, fc4_units=450):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        #self.fc3 = nn.Linear(fc2_units, fc3_units)
        #self.fc4 = nn.Linear(fc3_units, fc4_units)
        self.fc5 = nn.Linear(fc4_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        #self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        #self.fc4.weight.data.uniform_(*hidden_init(self.fc4))
        self.fc5.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        #x = F.relu(self.fc4(x))
        return self.fc5(x)
