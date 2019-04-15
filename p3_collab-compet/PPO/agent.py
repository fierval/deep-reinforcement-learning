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

    def surrogate(self, old_log_probs, states, actions, advantages):
        """
        Training step for a # of epochs
        """

        rewards_normalized = (advantages - torch.mean(advantages)) / (torch.std(advantages) + 1.e-10)

        # convert states to policy (or probability)
        _, new_log_probs, entropy = self.policy(states, actions)

        # ratio for clipping
        ratio = torch.exp(new_log_probs - old_log_probs)

        # clipped function
        clip = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon)
        clipped_surrogate = torch.min(ratio*rewards_normalized, clip*rewards_normalized)
        clipped_surrogate_mean = torch.mean(clipped_surrogate)
        entropy_mean = self.beta * torch.mean(entropy)

        # this returns an average of all the entries of the tensor
        # effective computing L_sur^clip / T
        # averaged over time-step and number of trajectories
        # this is desirable because we have normalized our rewards
        return clipped_surrogate_mean + entropy_mean

    def learn(self, old_log_probs, states, actions, advantages, returns):
        """Learning step
        
        """

        # surrogate function (actor)
        self.optimizer.zero_grad()

        L = - self.surrogate(old_log_probs, states, actions, advantages)

        L.backward()
        
        self.optimizer.step()

        self.tb_tracker.track(f"loss_policy", L.to("cpu"), self.t_step)
        del L

        # v-function (critic)
        self.optimizer_critic.zero_grad()

        values = self.policy_critic(states)
        loss_values = F.mse_loss(values, returns)
        loss_values.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_critic.parameters(), 10.)
        
        self.optimizer_critic.step()
        self.tb_tracker.track(f"loss_value", loss_values.to("cpu"), self.t_step)

        # decay epsilon and beta as we train
        #self.epsilon *= 0.9999
        #self.beta *= 0.9995
        self.t_step += 1

        