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

    def __init__(self, input_channels, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.Conv1 = nn.Conv2d(input_channels, out_channels=256, kernel_size=8, stride=4)
        self.Conv2 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=4, stride=2)
        self.Conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(896, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.fcout = nn.Linear(32, action_size)

        self.fc1_val = nn.Linear(896, 128)
        self.relu1_val = nn.ReLU()
        self.fc2_val = nn.Linear(128, 64)
        self.relu2_val = nn.ReLU()
        self.fc3_val = nn.Linear(64, 32)
        self.relu3_val = nn.ReLU()
        self.fcout_val = nn.Linear(32, 1)

    
    def forward(self, state):
        """Build a network that maps state -> action values."""

        state = F.relu(self.Conv1(state))
        state = F.relu(self.Conv2(state))
        state = F.relu(self.Conv3(state))
        state = torch.flatten(state, start_dim=1)
        #print("SHAPE", state.shape)

        advantage = self.relu1(self.fc1(state))
        advantage = self.relu2(self.fc2(advantage))
        advantage = self.relu3(self.fc3(advantage))
        advantage = self.fcout(advantage)

        value = self.relu1(self.fc1(state))
        value = self.relu2(self.fc2(value))
        value = self.relu3(self.fc3(value))
        value = self.fcout(value)

        mean_advantage = advantage.mean(1).unsqueeze(1).expand_as(advantage)
        value = value.expand_as(advantage)
        
        Q = value + advantage - mean_advantage

        return Q
        
