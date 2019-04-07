import torch
import torch.nn as nn

def SharedPolicy(state_size, action_size, seed = 1, fc1_units=128, fc2_units=128):
    """Initialize parameters and build model.
    Params
    ======
        state_size (int): Dimension of each state
        action_size (int): Dimension of each action
        index (int): index input
        seed (int): Random seed
        fc1_units (int): Number of nodes in first hidden layer
        fc2_units (int): Number of nodes in second hidden layer
    """
    
    torch.manual_seed(seed)

    # add 1 to state size for agent idx
    model = torch.nn.Sequential(
        nn.Linear(state_size + 1, fc1_units),
        nn.ReLU(),
        nn.Linear(fc1_units, fc2_units),
        nn.ReLU(),
        nn.Linear(fc2_units, action_size),
        nn.Tanh()
    )

    return model
