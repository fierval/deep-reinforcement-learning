'''Models courtesy of https://github.com/Shmuma/Deep-Reinforcement-Learning-Hands-On
'''

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_units=256):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc_units)
        self.fc2 = nn.Linear(fc_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        return F.tanh(self.fc2(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=256, fc2_units=256, fc3_units=128):
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
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.leaky_relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.fc4(x)

class DDPGActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super().__init__()

        self.hiden_1_len = 256
        self.hidden_2_len = 128

        self.net = nn.Sequential(
            nn.Linear(obs_size, self.hiden_1_len),
            nn.ReLU(),
            nn.Linear(self.hiden_1_len, self.hidden_2_len),
            nn.ReLU(),
            nn.Linear(self.hidden_2_len, act_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

class D4PGCritic(nn.Module):
    '''D4PGCritic model using a parameterized distribution described in the paper:
    https://arxiv.org/pdf/1707.06887.pdf (Section 4)
    '''
    def __init__(self, obs_size, act_size, n_atoms, v_min, v_max):
        super().__init__()

        self.hiden_1_len = 256
        self.hidden_2_len = 128

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, self.hiden_1_len),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(self.hiden_1_len + act_size, self.hidden_2_len),
            nn.ReLU(),
            nn.Linear(self.hidden_2_len, n_atoms)
        )

        delta = (v_max - v_min) / (n_atoms - 1)
        self.register_buffer("supports", torch.arange(v_min, v_max + delta, delta))

    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))

    def distr_to_q(self, distr):
        weights = F.softmax(distr, dim=1) * self.supports
        res = weights.sum(dim=1)
        return res.unsqueeze(dim=-1)
