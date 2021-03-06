"""Auxiliary classes and functions from https://github.com/Shmuma/Deep-Reinforcement-Learning-Hands-On

"""

import sys
import time
import operator
from datetime import timedelta
import numpy as np
import collections
import copy

import torch
import torch.nn as nn

def distr_projection(next_distr_v, rewards_v, dones_mask_t, gamma, device, n_atoms, delta_z, vmin, vmax):
    '''Projects one parameterized distribution onto another given the target distribution number of intervals
        min/max values and detla (interval size)
    
    Arguments:
        next_distr_v {Tensor} -- distro to project
        rewards_v {array-like} -- rewards to use for bellman equation
        dones_mask_t {Tensor} -- completed trajectory mask (array of True's set at indices of completed trajectories)
        gamma {float} -- discount
        device {string} -- cuda or cpu
        n_atoms {int} -- numper of intervals
        delta_z {float} -- interval size: (Vmax - Vmin) / (n_atoms - 1)
        vmin {float} -- min interval
        vmax {float} -- max interval
    
    Returns:
        FloatTensor -- projected distribution
    '''

    next_distr = next_distr_v.data.cpu().numpy()
    dones_mask = dones_mask_t.data.cpu().numpy().astype(np.bool)
    rewards = rewards_v.data.cpu().numpy()

    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, n_atoms), dtype=np.float32)

    for atom in range(n_atoms):
        tz_j = np.minimum(vmax, np.maximum(vmin, rewards + (vmin + atom * delta_z) * gamma))
        b_j = (tz_j - vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]

    if dones_mask.any():
        proj_distr[dones_mask] = 0.0
        tz_j = np.minimum(vmax, np.maximum(vmin, rewards[dones_mask]))
        b_j = (tz_j - vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        eq_dones = dones_mask.copy()
        eq_dones[dones_mask] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u != l
        ne_dones = dones_mask.copy()
        ne_dones[dones_mask] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]
    return torch.FloatTensor(proj_distr).to(device)

class TBMeanTracker:
    """
    TensorBoard value tracker: allows to batch fixed amount of historical values and write their mean into TB

    Designed and tested with pytorch-tensorboard in mind
    """
    def __init__(self, writer, batch_size):
        """
        :param writer: writer with close() and add_scalar() methods
        :param batch_size: integer size of batch to track
        """
        assert isinstance(batch_size, int)
        assert writer is not None
        self.writer = writer
        self.batch_size = batch_size
        self._batches = collections.defaultdict(list)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()

    @staticmethod
    def _as_float(value):
        assert isinstance(value, (float, int, np.ndarray, np.generic, torch.autograd.Variable)) or torch.is_tensor(value)
        tensor_val = None
        if isinstance(value, torch.autograd.Variable):
            tensor_val = value.data
        elif torch.is_tensor(value):
            tensor_val = value

        if tensor_val is not None:
            return tensor_val.float().mean()
        elif isinstance(value, np.ndarray):
            return float(np.mean(value))
        else:
            return float(value)

    def track(self, param_name, value, iter_index):
        assert isinstance(param_name, str)
        assert isinstance(iter_index, int)

        data = self._batches[param_name]
        data.append(self._as_float(value))

        if len(data) >= self.batch_size:
            self.writer.add_scalar(param_name, np.mean(data), iter_index)
            data.clear()

class RewardTracker:
    def __init__(self, writer, mean_window = 100):
        self.writer = writer
        self.total_rewards = []
        self.mean_window = mean_window

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, scores, reward, frame, duration, epsilon=None):
        self.total_rewards.append(reward)
        i_episode = len(self.total_rewards)
        mean_reward = np.mean(self.total_rewards[-self.mean_window :])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: reward %.3f, mean reward %.3f, min %.3f, max %.3f, duration %.2f s" % (
            i_episode, reward, mean_reward, np.min(scores), np.max(scores), duration))
        sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        self.writer.add_scalar("min_reward", np.min(scores), frame)
        self.writer.add_scalar("max_reward", np.max(scores), frame)
        self.writer.add_scalar("duration", duration, frame)
        
        return mean_reward if len(self.total_rewards) > 30 else None
class TargetNet:
    """
    Wrapper around model which provides copy of it instead of trained weights
    """
    def __init__(self, model):
        self.model = model
        self.target_model = model
        self.target_model.load_state_dict(model.state_dict())

    def sync(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def alpha_sync(self, alpha):
        """
        Blend params of target net with params from the model
        :param alpha:
        """
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        self.target_model.load_state_dict(tgt_state)
