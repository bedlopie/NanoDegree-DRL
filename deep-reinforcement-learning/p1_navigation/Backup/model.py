import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

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
        dropout = [0.0, 0.0, 0.0, 0.0]
        
        # Modify the end of the model to match our need, so replacing the classifier definition
        module_ordered_dict = []
        if dropout[0] != 0.0:
            module_ordered_dict.append(('dropout{}'.format(1), nn.Dropout(p=dropout[0])))
        module_ordered_dict.append(('fc1', nn.Linear(state_size, hidden_layers[0])))
        
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        for i, (h1, h2) in enumerate(layer_sizes, 2):
            module_ordered_dict.append(('relu{}'.format(i), nn.ReLU()))
            if dropout[i-1] != 0.0:
                module_ordered_dict.append(('dropout{}'.format(i), nn.Dropout(p=dropout[i-1])))
            module_ordered_dict.append(('fc{}'.format(i), nn.Linear(h1, h2)))
            
        module_ordered_dict.append(('reluout', nn.ReLU()))
        module_ordered_dict.append(('dropout', nn.Dropout(p=dropout[-1])))
        module_ordered_dict.append(('fcout', nn.Linear(hidden_layers[-1], action_size)))
        
        self.banana_model = nn.Sequential(OrderedDict(module_ordered_dict))

    
    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        return self.banana_model.forward(state)
