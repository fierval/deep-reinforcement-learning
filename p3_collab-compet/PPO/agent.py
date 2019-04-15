import numpy as np
import random
import torch
import torch.nn.functional as F
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPOAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, policy, optimizer, policy_critic, optimizer_critic, tb_tracker, epsilon, beta):
        """Initialize an Agent object.
        
        Params
        ======
            policy (Pytorch network): policy to be learned/executed
            optimizer (Pytorch optimizer): optimizer to be used
            policy_critic (Pytorch network): policy critic for V function
            optimizer_critic (Pytorch optimizer): optimizer for crtic
            tb_tracker (tensorboard tracker)
            epsilon - action clipping: [1 - epsilon, 1 + epsilon]
            beta - regularization parameter
        """
        
        self.policy = policy
        self.optimizer = optimizer
        self.policy_critic = policy_critic
        self.optimizer_critic = optimizer_critic
        self.tb_tracker = tb_tracker
        
        self.beta = beta
        self.epsilon = epsilon
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def act(self, state, idx):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        idx = torch.tensor([idx], dtype=torch.float32, device=device).unsqueeze(0)

        state = torch.cat((state, idx), dim=1)

        self.policy.eval()
        with torch.no_grad():
            actions, _, _ = self.policy(state)
        self.policy.train()

        return torch.clamp(actions, -1, 1)

    def learn(self, old_log_probs, states, actions, advantages, returns):
        """Learning step
        
        """
        rewards_normalized = (advantages - torch.mean(advantages)) / (torch.std(advantages) + 1.e-10)

        _, log_probs, entropy = self.policy(states, actions)

        values = self.policy_critic(states)

        # critic loss
        self.optimizer_critic.zero_grad()
        loss_values = F.mse_loss(values, returns)

        loss_values.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_critic.parameters(), 10.)

        self.optimizer_critic.step()
        self.tb_tracker.track(f"loss_values", loss_values.to("cpu"), self.t_step)

        # actor loss
        self.optimizer.zero_grad()
        ratio = torch.exp(log_probs - old_log_probs)
        ratio_clamped = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        adv_PPO = torch.min(ratio * rewards_normalized, ratio_clamped * rewards_normalized)

        loss_policy = -torch.mean(adv_PPO + self.beta * entropy)
        loss_policy.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.)
        self.optimizer.step()

        # surrogate function (actor)

        self.tb_tracker.track(f"loss_policy", loss_policy.to("cpu"), self.t_step)
        del loss_policy

        # decay epsilon and beta as we train
        #self.epsilon *= 0.9999
        #self.beta *= 0.9995
        self.t_step += 1

        