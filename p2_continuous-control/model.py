'''Models courtesy of https://github.com/Shmuma/Deep-Reinforcement-Learning-Hands-On
'''

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class DDPGActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, act_size),
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

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(400 + act_size, 300),
            nn.ReLU(),
            nn.Linear(300, n_atoms)
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
