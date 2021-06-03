import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict


class Residual_Block(nn.Module):

    def __init__(self, channels_in, channels_out):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Residual_Block, self).__init__()

        self.channels_in = channels_in
        self.channels_out = channels_out
        self.conv_a = nn.Conv2d(channels_in, channels_out, 3, padding=1)
        self.bn_a = nn.BatchNorm2d(channels_out)
        self.relu_a = nn.ReLU()
        self.conv_b = nn.Conv2d(channels_out, channels_out, 3, padding=1)
        self.bn_b = nn.BatchNorm2d(channels_out)
        self.relu_b = nn.ReLU()
        self.conv1x1 = nn.Conv2d(channels_in, channels_out, 1)

    
    def forward(self, state):
        
        residual = state
        state = self.conv_a(state)
        state = self.bn_a(state)
        state = self.relu_a(state)
        state = self.conv_b(state)
        state = self.bn_b(state)
        if self.channels_in != self.channels_out:
            residual = self.conv1x1(residual)
        state += residual
        state = self.relu_b(state)

        return state


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        hidden_layers = [128, 64, 32] # best = [128, 64, 32]
        in_channels = 128
        channels_conv = 32

        # Addition on the CNN layers

        # First Convolution

        self.conv_in = nn.Conv2d(12, in_channels, 3, padding=1)
        self.bn_in = nn.BatchNorm2d(in_channels)
        self.relu_in = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)

        # Multiple Residuals layers
        self.residual_1 = Residual_Block(in_channels, channels_conv)
        #self.residual_2 = Residual_Block(channels_conv, channels_conv)
        #self.residual_3 = Residual_Block(channels_conv, channels_conv)
        #self.residual_4 = Residual_Block(channels_conv, channels_conv)
        #self.residual_5 = Residual_Block(channels_conv, channels_conv)

        self.maxpool2 = nn.MaxPool2d(2, 2)
        # flatten size
        flatten_size = channels_conv * 21 * 21

        # Fully connected layers
        # self.fc1 = nn.Linear(flatten_size, hidden_layers[0])
        # self.relu1 = nn.ReLU()
        # self.fc2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        # self.relu2 = nn.ReLU()
        #self.fc3 = nn.Linear(hidden_layers[1], hidden_layers[2])
        #self.relu3 = nn.ReLU()
        # self.fcout = nn.Linear(hidden_layers[2], action_size)

        self.fc1 = nn.Linear(flatten_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 32)
        self.relu2 = nn.ReLU()
        #self.fc3 = nn.Linear(64, 32)
        #self.relu3 = nn.ReLU()
        self.fcout = nn.Linear(32, action_size)




    
    def forward(self, state):
        """Build a network that maps state -> action values."""

        #print("Start forward shape", state.shape)
        #print(state)
        state = self.conv_in(state)
        state = self.bn_in(state)
        state = self.relu_in(state)
        state = self.maxpool(state) 
        state = self.residual_1(state)
        #state = self.residual_2(state)
        #state = self.residual_3(state)
        #state = self.residual_4(state)
        #state = self.residual_5(state)
        state = self.maxpool2(state)
        #print("After conv shape", state.shape)
        state = torch.flatten(state, start_dim=1)
        #print("After flat shape", state.shape)
        state = self.fc1(state)
        state = self.relu1(state)
        state = self.fc2(state)
        state = self.relu2(state)
        #state = self.fc3(state)
        #state = self.relu3(state)
        state = self.fcout(state)

        
        return state
        
