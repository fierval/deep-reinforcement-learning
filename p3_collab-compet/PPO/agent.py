import numpy as np
import random
import torch
import torch.nn.functional as F
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPOAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, index, policy, optimizer, policy_critic, optimizer_critic, tb_tracker, epsilon, beta):
        """Initialize an Agent object.
        
        Params
        ======
            index (int): agent index parameter
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
        
        # extra parameter to be added to states
        self.idx_me = torch.tensor([index]).unsqueeze(0).to(device)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def calc_logprob(self, dist, actions_v):
        logprob = dist.log_prob(actions_v).sum(-1).unsqueeze(-1)
        return logprob

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.policy.eval()
        with torch.no_grad():
            _, dist = self.policy(state, self.idx_me)
        self.policy.train()

        return torch.clamp(dist.sample(), -1, 1)

    def surrogate(self, old_log_probs, states, actions, rewards_normalized):
        """
        Training step for a # of epochs
        """
      
        # convert everything into pytorch tensors and move to gpu if available
        actions = torch.tensor(actions, dtype=torch.int8, device=device)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float, device=device)
        rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)

        # convert states to policy (or probability)
        _, dist = self.policy(states, self.idx_me)
        new_log_probs = self.calc_logprob(dist, actions)

        # ratio for clipping
        ratio = torch.exp(new_log_probs - old_log_probs)

        # clipped function
        clip = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon)
        clipped_surrogate = torch.min(ratio*rewards, clip*rewards)

        entropy = dist.entropy().sum(-1).unsqueeze(-1)

        # this returns an average of all the entries of the tensor
        # effective computing L_sur^clip / T
        # averaged over time-step and number of trajectories
        # this is desirable because we have normalized our rewards
        return torch.mean(clipped_surrogate + self.beta*entropy)

    def learn(self, old_log_probs, states, actions, advantages, returns, values):
        """[summary]
        
        Arguments:
            old_log_probs {[type]} -- log probabilities
            states {[type]} -- states
            actions {[type]} -- actions
            rewards {[type]} -- rewards
            dones {[type]} -- dones
        """

        future_rewards = (advantages - torch.mean(advantages)) / (torch.std(advantages) + 1.e-10)
        
        # v-function (critic)
        self.optimizer_critic.zero_grad()

        loss_values = F.mse_loss(values, returns)
        loss_values.backward()
        self.optimizer_critic.step()

        # surrogate function (actor)
        L = - self.surrogate(old_log_probs, states, actions, future_rewards)
        
        self.optimizer.zero_grad()
        L.backward()
        self.optimizer.step()
        del L
        
        # decay epsilon and beta as we train
        # self.epsilon *= 0.999
        # self.beta *= 0.995
