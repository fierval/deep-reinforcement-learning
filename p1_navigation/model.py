import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

def get_seed(seed):
    return torch.manual_seed(datetime.now().timestamp() if seed is None else seed) 

def QNetwork(state_size, action_size, seed = None, fc1_units=256, fc2_units=256):
    """Initialize parameters and build model.
    Params
    ======
        state_size (int): Dimension of each state
        action_size (int): Dimension of each action
        seed (int): Random seed
        fc1_units (int): Number of nodes in first hidden layer
        fc2_units (int): Number of nodes in second hidden layer
    """
    
    seed = get_seed(seed)
    model = torch.nn.Sequential(
        nn.Linear(state_size, fc1_units),
        nn.ReLU(),
        nn.Linear(fc1_units, fc2_units),
        nn.ReLU(),
        nn.Linear(fc2_units, action_size)
    )

    return model

class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed = None, fc1_size = 128, fc2_size = 128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        
        super().__init__()

        self.num_actions = action_size

        self.seed = get_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        fc_adv_hidden_size = fc_val_hidden_size = 32

        ## Here we separate into two streams
        # The one that calculate V(s)
        self.fc_val_hidden = nn.Linear(fc2_size, fc_val_hidden_size)
        self.fc_val = nn.Linear(fc_val_hidden_size, 1)
        
        # The one that calculate A(s,a)
        self.fc_adv_hidden = nn.Linear(fc2_size, fc_adv_hidden_size)
        self.fc_adv = nn.Linear(fc_adv_hidden_size, action_size)



    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        val = F.relu(self.fc_val_hidden(x))
        val = self.fc_val(val)

        adv = F.relu(self.fc_adv_hidden(x))
        adv = self.fc_adv(adv)

        # Q(s,a) = V(s) + (A(s,a) - max A(s, a))
        qsa = val + adv - adv.max(1)[0].unsqueeze(1).expand(state.size(0), self.num_actions)
        return qsa
