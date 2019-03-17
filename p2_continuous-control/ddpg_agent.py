import numpy as np
import random
import copy, math
from collections import namedtuple, deque

from model import DDPGActor, D4PGCritic
from memory import PrioritizedReplayBuffer
from utils import distr_projection, RewardTracker, TBMeanTracker

from ptan import 
import torch
import torch.nn.functional as F
import torch.optim as optim

# use tensorboard to monitor progress
from tensorboardX import SummaryWriter

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0.0001   # L2 weight decay
REWARD_STEPS = 1        # TODO: look-ahead steps. For now this can only be set to 1.

ALPHA = 0.8             # priority exponent for prioritized replacement
BETA = 0.7              # initial beta (annealed to 1) for prioritized replacement

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# support parameters for the parameterized distributed Q-function
Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)

class D4PGAgent():
    """D4PG agent implementation: https://openreview.net/pdf?id=SyZipzbCb"""
    
    def __init__(self, num_agents, update_fraction, state_size, action_size, random_seed, log_name, anneal_over = 1e5):
        """Initialize an Agent object.
        
        Params
        ======
            num_agens (int): number of agents generating states/actions
            update_fraction (float): how often to update
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            log_name (string): tensorboard log suffix
            anneal_over (int): anneal beta to 1 for priority replacement over these many steps
        """

        self.num_agents = num_agents
        self.state_size = state_size
        self.action_size = action_size
        assert 0 < update_fraction <= 1., "update_fraction should be in (0, 1]"

        update_every = int(math.ceil(update_fraction * num_agents))
        assert update_every > 0, "update_every too small"

        # indicies along which the arrays of states, actions, etc will be split
        # so we send them to training every index that occurs in this array
        self.learning_step_idxs = set(np.arange(num_agents, step_size=update_every))

        random.seed(random_seed)
        torch.manual_seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = DDPGActor(state_size, action_size).to(device)
        self.actor_target = DDPGActor(state_size, action_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = D4PGCritic(state_size, action_size, N_ATOMS, Vmin, Vmax).to(device)
        self.critic_target = D4PGCritic(state_size, action_size, N_ATOMS, Vmin, Vmax).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self.anneal_beta = (1. - BETA) / anneal_over

        # Replay memory with action clipping to -1, 1
        self.memory = PrioritizedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed, ALPHA, BETA, clip_action=True)
    
        # Tensorboard interface
        self.writer = SummaryWriter(comment=f"d4pg-{log_name}")
        self.tb_tracker = TBMeanTracker(self.writer, batch_size=10)
        self.step_t = 0

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward

        for i, (state, action, reward, next_state, done) in enumerate(zip(states, actions, rewards, next_states, dones)):
            self.memory.add(state, action, reward, next_state, done)
            self.step_t += 1

            # Learn, if enough samples are available in memory
            if len(self.memory) > BATCH_SIZE and i in self.learning_step_idxs:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        return np.clip(action, -1, 1)


    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value distribution

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done, experience_idxs_in_buffer, weights) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, idxs, weights = experiences

        # ---------------------------- update critic ---------------------------- #
        Q_expected_distribution = self.critic_local(states, actions)
        
        next_actions = self.actor_target(next_states)

        target_distribution_next = self.critic_target(next_states, next_actions)
        Q_target_distribution_next = F.softmax(target_distribution_next, dim=1)

        Q_target_distribution = distr_projection(Q_target_distribution_next, rewards, dones, 
            gamma ** REWARD_STEPS, device, N_ATOMS, DELTA_Z, Vmin, Vmax)

        prob_dist = -F.log_softmax(Q_expected_distribution, dim=1) * Q_target_distribution
        critic_loss = prob_dist.sum(dim=1).mean()

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        (critic_loss * weights).backward()
        self.critic_optimizer.step()

        # update replay weights
        q_expected = self.critic_local.distr_to_q(Q_expected_distribution)
        q_target = self.critic_target.distr_to_q(Q_target_distribution)
        self.memory.anneal_beta(self.anneal_beta)

        updates = torch.abs(q_expected - q_target).cpu().data.squeeze(1).numpy()
        self.memory.update_priorities(idxs, updates)

        self.tb_tracker.track("loss_critic", critic_loss.to("cpu"), self.step_t)

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)

        critic_distr = self.critic_local(states, actions_pred)

        actor_loss = -self.critic_local.distr_to_q(critic_distr).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.tb_tracker.track("loss_actor", actor_loss.to("cpu"), self.step_t)

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
