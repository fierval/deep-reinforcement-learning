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

    def forward(self, x, idx):
        x = torch.cat((x, idx), dim = 1)

        mean = self.mu(x)
        dist = torch.distributions.Normal(mean, F.softplus(self.logstd))
        return mean, dist

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

