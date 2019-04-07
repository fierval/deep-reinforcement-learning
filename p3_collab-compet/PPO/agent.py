import numpy as np
import random
import torch

GAMMA = 0.99            # discount factor
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network
BETA = 0.01             # regularization parameter for entropy term
EPSILON = 0.1           # action clipping param: [1-EPSILON, 1+EPSILON]
EPOCHS = 4              # train for this number of epochs at a time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, index, policy, optimizer):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            index (int): agent index parameter
            policy (Pytorch network): policy to be learned/executed
            optimizer (Pytorch optimizer): optimizer to be used
        """
        self.state_size = state_size
        self.action_size = action_size

        # self.policy = Policy(state_size, action_size, seed).to(device)
        # self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.policy = policy
        self.optimizer = optimizer

        self.beta = BETA
        self.epsilon = EPSILON
        self.epochs = EPOCHS
        
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

    def act(self, state):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.policy.eval()
        with torch.no_grad():
            action_values = self.policy_func(state)
        self.policy.train()
        return action_values

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
        new_probs = self.policy_func(states)
        
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
