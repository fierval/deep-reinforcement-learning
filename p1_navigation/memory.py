import torch
from collections import deque, namedtuple
import random
import numpy as np

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, alpha, beta, num_episodes):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            alpha (float) < 1: exponent for priorities computation
            beta (float): weight bias correction coefficient
            num_episodes (int): number of episodes required for beta annealing to 1.0
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        # priorities buffer
        self.priorities = np.ones(buffer_size).astype(np.float)

        self.alpha = alpha
        self.beta = beta
        self.anneal = 0.0

        if self.beta < 1.0:
            self.anneal = (1.0 - self.beta) / float(num_episodes)

        self.buffer_size = buffer_size

        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample_experiences(self):
        '''
        Sample experiences based on https://arxiv.org/pdf/1511.05952.pdf: Prioritized Experience Replay
        '''
        # update priorities for sampling
        priorities = np.power(self.priorities, self.alpha)
        priorities = priorities / np.sum(self.priorities)

        # get experiences based on priorities
        idx_priorities = np.random.choice(range(self.buffer_size), size= self.batch_size, p=priorities)
        probs = np.take(priorities, idx_priorities)

        return idx_priorities, probs

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experience_idxs, probs = self.sample_experiences()
        experiences = [self.memory[e] for e in experience_idxs]

        weights = np.power(1. / self.buffer_size * probs, self.beta)
        weights = torch.from_numpy(weights / np.max(weights)).float().to(self.device)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones, experience_idxs, weights)

    def update_priorities(self, idxs, priorities):
        # anneal beta for next episode
        self.beta += self.anneal
        self.priorities[idxs] = priorities

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)