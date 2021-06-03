import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, Dueling):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.Dueling = Dueling
        
        hidden_layers = [128, 64, 32] 
        
        self.fc1 = nn.Linear(state_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        self.fcout = nn.Linear(32, action_size)

        if Dueling:
            self.fc1_val = nn.Linear(state_size, 128)
            self.relu1_val = nn.ReLU()
            self.fc2_val = nn.Linear(128, 64)
            self.relu2_val = nn.ReLU()
            self.fc3_val = nn.Linear(64, 32)
            self.relu3_val = nn.ReLU()
            self.fcout_val = nn.Linear(32, 1)
    
    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        advantage = self.relu1(self.fc1(state))
        advantage = self.relu2(self.fc2(advantage))
        advantage = self.relu3(self.fc3(advantage))
        advantage = self.fcout(advantage)

        if self.Dueling:
            value = self.relu1(self.fc1(state))
            value = self.relu2(self.fc2(value))
            value = self.relu3(self.fc3(value))
            value = self.fcout_val(value)

            mean_advantage = advantage.mean(1).unsqueeze(1).expand_as(advantage)
            value = value.expand_as(advantage)
            
            Q = value + advantage - mean_advantage

            return Q
        return advantage
