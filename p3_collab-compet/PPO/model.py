import torch
import torch.nn as nn
import torch.nn.functional as F

HID_SIZE = 64

class GaussianPolicyActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super().__init__()

        self.action_dim = act_size
        self.state_dim = obs_size

        self.mu = nn.Sequential(
            nn.Linear(obs_size + 1, HID_SIZE),
            nn.Tanh(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.Tanh(),
            nn.Linear(HID_SIZE, act_size),
            nn.Tanh(),
        )
        self.logstd = nn.Parameter(torch.zeros(act_size))

    def forward(self, x, idx, actions=None):
        """
        Inspired by: https://github.com/tnakae/Udacity-DeepRL-p3-collab-compet/blob/master/PPO/network.py
        
        Arguments:
            x {tensor} -- state
            idx {tensor} -- agent id
        
        Keyword Arguments:
            actions {tensor} -- [actions, if any] (default: {None})
        
        Returns:
            [tensor] -- [actions, log_prob, entropy, distribution]
        """

        x = torch.cat((x, idx), dim = 1)

        mean = self.mu(x)
        dist = torch.distributions.Normal(mean, F.softplus(self.logstd))

        if actions is None:
            actions = dist.sample()
        log_prob = dist.log_prob(actions)
        log_prob = torch.sum(log_prob, dim=-1)
        return actions, log_prob, dist

class ModelCritic(nn.Module):
    def __init__(self, obs_size):
        super().__init__()

        self.value = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1),
        )

    def forward(self, x):
        return self.value(x)

