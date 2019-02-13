import torch
import torch.nn as nn
from datetime import datetime

def QNetwork(state_size, action_size, seed = None, fc1_units=64, fc2_units=64):
    """Initialize parameters and build model.
    Params
    ======
        state_size (int): Dimension of each state
        action_size (int): Dimension of each action
        seed (int): Random seed
        fc1_units (int): Number of nodes in first hidden layer
        fc2_units (int): Number of nodes in second hidden layer
    """
    
    seed = torch.manual_seed(datetime.now().timestamp() if seed is None else seed) 
    model = torch.nn.Sequential(
        nn.Linear(state_size, fc1_units),
        nn.ReLU(),
        nn.Linear(fc1_units, fc2_units),
        nn.ReLU(),
        nn.Linear(fc2_units, action_size)
    )

    return model
