import torch
from collections import deque, namedtuple
import random
import numpy as np

class PrioritizedReplayBuffer:
    """Fixed-size buffer to store experience tuples.
    Implements prioritized experience replay
    """

    def __init__(self, action_size, buffer_size, batch_size, seed, alpha, beta, anneal_over):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            alpha (float) < 1: exponent for priorities computation
            beta (float): weight bias correction coefficient
            anneal (int): linear annealing factor for beta
            clip_action (bool): should we clip action to [-1, 1]
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        # priorities buffer
        self.priorities = np.ones(buffer_size).astype(np.float)

        self.alpha = alpha
        self.beta = beta

        self.anneal = (1. - beta) / anneal_over
        self.buffer_size = buffer_size

        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        buffer_size = len(self.memory)
        self.priorities[buffer_size - 1] = np.max(self.priorities[:buffer_size])
    
    def sample_experiences(self):
        '''
        Sample experiences based on https://arxiv.org/pdf/1511.05952.pdf: Prioritized Experience Replay
        '''
        # update priorities for sampling
        buffer_size = len(self.memory)

        priorities = self.priorities[:buffer_size]
        priorities = np.power(priorities, self.alpha)
        priorities = priorities / np.sum(priorities)

        # get experiences based on priorities
        idx_priorities = np.random.choice(range(buffer_size), size=self.batch_size, p=priorities)
        probs = np.take(priorities, idx_priorities)

        return idx_priorities, probs

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experience_idxs, probs = self.sample_experiences()
        experiences = [self.memory[e] for e in experience_idxs]

        weights = np.power(self.buffer_size * probs, -self.beta)
        weights = torch.from_numpy(weights / np.max(weights)).float().to(self.device)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        self.anneal_beta()

        return (states, actions, rewards, next_states, dones, experience_idxs, weights)

    def update_priorities(self, idxs, priorities):
       
        self.priorities[idxs] = priorities

    def anneal_beta(self):
        self.beta = min(self.beta + self.anneal, 1.)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)        