import numpy as np
import random
import torch
import torch.nn.functional as F
import math

GAMMA = 0.99            # discount factor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPOAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, index, policy, optimizer, epochs, epsilon, beta):
        """Initialize an Agent object.
        
        Params
        ======
            index (int): agent index parameter
            policy (Pytorch network): policy to be learned/executed
            optimizer (Pytorch optimizer): optimizer to be used
            epsilon - action clipping: [1 - epsilon, 1 + epsilon]
            beta - regularization parameter
            epochs - number of training epochs
        """
        # self.policy = Policy(state_size, action_size, seed).to(device)
        # self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.policy = policy
        self.optimizer = optimizer

        self.beta = beta
        self.epsilon = epsilon
        self.epochs = epochs
        
        # extra parameter to be added to states
        self.idx_me = None
        self.index = index

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def get_state_tensor(self, state):
        if self.idx_me is None:
            self.idx_me = torch.ones(state.shape[0]).unsqueeze(1).to(device) * self.index

        return torch.cat((state, self.idx_me), dim = 1)

    def policy_func(self, state):
        """
        Wrapper for running through policy. We need to add the index parameter
        """
        st = self.get_state_tensor(state)
        return self.policy(st)

    def calc_logprob(self, mu_v, actions_v):
        logstd_v = self.policy.logstd
        p1 = - ((mu_v - actions_v) ** 2) / (2*torch.exp(logstd_v).clamp(min=1e-3))
        p2 = - torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd_v)))
        return p1 + p2

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.policy.eval()
        with torch.no_grad():
            action = self.policy_func(state)
        self.policy.train()
        return action

    def act_train(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.policy.eval()
        with torch.no_grad():
            mu_v = self.policy_func(state)
            mu = mu_v.data.cpu().numpy()
            logstd = self.policy.logstd.data.cpu().numpy()
            action = mu + np.exp(logstd) * np.random.normal(size=logstd.shape)
            action = np.clip(action, -1, 1)
        self.policy.train()
        return action

    def surrogate(self, old_probs, states, actions, rewards):
        """
        Training step for a # of epochs
        """
        discount = GAMMA ** np.arange(len(rewards))
        rewards = np.asarray(rewards)*discount[:,np.newaxis]
        
        # convert rewards to future rewards
        rewards_future = rewards[::-1].cumsum(axis=0)[::-1]
        
        mean = np.mean(rewards_future, axis=1)
        std = np.std(rewards_future, axis=1) + 1.0e-10

        rewards_normalized = (rewards_future - mean[:,np.newaxis])/std[:,np.newaxis]

        # convert everything into pytorch tensors and move to gpu if available
        actions = torch.tensor(actions, dtype=torch.int8, device=device)
        old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
        rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)

        # convert states to policy (or probability)
        mu_v = self.policy_func(states)
        new_log_probs = self.calc_logprob(mu_v, actions)
        new_probs = torch.exp(new_log_probs)

        # ratio for clipping
        ratio = new_probs/old_probs

        # clipped function
        clip = torch.clamp(ratio, 1-self.epsilon, 1+self.epsilon)
        clipped_surrogate = torch.min(ratio*rewards, clip*rewards)

        # include a regularization term
        # this steers new_policy towards 0.5
        # add in 1.e-10 to avoid log(0) which gives nan
        entropy = -(new_probs*torch.log(old_probs+1.e-10)+ \
            (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))
        
        # this returns an average of all the entries of the tensor
        # effective computing L_sur^clip / T
        # averaged over time-step and number of trajectories
        # this is desirable because we have normalized our rewards
        return torch.mean(clipped_surrogate + self.beta*entropy)

    def learn(self, old_probs, states, actions, rewards):
        """Run the surrogate function for a few epochs

        Params
        ======
            trajectories (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
        """

        for _ in range(self.epochs):
            L = - self.surrogate(old_probs, states, actions, rewards)
            
            self.optimizer.zero_grad()
            L.backward()
            self.optimizer.step()
            del L
        
        # decay epsilon and beta as we train
        self.epsilon *= 0.999
        self.beta *= 0.995
