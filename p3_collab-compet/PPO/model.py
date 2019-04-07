import torch
import torch.nn as nn
import torch.nn.functional as F

HID_SIZE = 64

class GaussianPolicy(nn.Module):
    def __init__(self, obs_size, act_size, seed=1):
        super().__init__()

        torch.manual_seed(seed)
        self.mu = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.Tanh(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.Tanh(),
            nn.Linear(HID_SIZE, act_size),
            nn.Tanh(),
        )
        self.logstd = nn.Parameter(torch.zeros(act_size))

    def forward(self, x):
        mean = self.mu(x)
        dist = torch.distributions.Normal(mean, F.softplus(self.logstd))
        return mean, dist
