import torch
import torch.nn as nn
import torch.nn.functional as F

HID_SIZE = 512

def xavier(sequential):
    for layer in sequential:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight.data)

class GaussianPolicyActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super().__init__()

        self.action_dim = act_size
        self.state_dim = obs_size

        hid_size = HID_SIZE
        hid_size_1 = HID_SIZE // 2

        self.mu = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.LeakyReLU(),
            nn.Linear(hid_size, hid_size_1),
            nn.LeakyReLU(),
            nn.Linear(hid_size_1, act_size),
            nn.Tanh(),
        )

        xavier(self.mu)                
        self.logstd = nn.Parameter(torch.zeros(act_size))

    def forward(self, x, actions=None):
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

        mean = self.mu(x)
        dist = torch.distributions.Normal(mean, F.softplus(self.logstd))

        if actions is None:
            actions = dist.sample()
        log_prob = dist.log_prob(actions)
        log_prob = torch.sum(log_prob, dim=-1)
        entropy = dist.entropy()
        entropy = torch.sum(entropy, dim=-1)

        return actions, log_prob, entropy

class ModelCritic(nn.Module):
    def __init__(self, obs_size):
        super().__init__()

        hid_size = HID_SIZE
        hid_size_1 = HID_SIZE // 2

        self.value = nn.Sequential(
            nn.Linear(obs_size, hid_size),
            nn.LeakyReLU(),
            nn.Linear(hid_size, hid_size_1),
            nn.LeakyReLU(),
            nn.Linear(hid_size_1, 1),
            nn.LeakyReLU(),
        )
        
        xavier(self.value)

    def forward(self, x):
        return self.value(x).squeeze(-1)

