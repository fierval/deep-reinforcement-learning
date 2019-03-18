import numpy as np
import random
import copy, math
from collections import namedtuple, deque

from utils import distr_projection, RewardTracker, TBMeanTracker

import torch
import torch.nn.functional as F
import torch.optim as optim

# use tensorboard to monitor progress
from d4pg_agent import D4PGAgent, N_EPISODES, MAX_T
from ddpg_agent import Agent

if __name__ == '__main__':
    from unityagents import UnityEnvironment  

    SOLVED_SCORE = 30
    MEAN_WINDOW = 100

    env = UnityEnvironment(file_name='p2_continuous-control/Reacher_Linux_1/Reacher.x86_64')

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True, )[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])
    
    n_episodes = N_EPISODES
    max_t = MAX_T

    agent = D4PGAgent(num_agents, 1, state_size, action_size, 1000)
    max_score = -np.Inf
    solved_episode = -np.Inf

    # tracks all the mean rewards etc
    with RewardTracker(agent.writer, MEAN_WINDOW) as reward_tracker:

        for i_episode in range(1, n_episodes+1):
            env_info = env.reset(train_mode=True)[brain_name]
            states = env_info.vector_observations
            scores = np.zeros(num_agents)

            for t in range(max_t):
                actions = agent.act(states)

                env_info = env.step(actions)[brain_name]           # send all actions to tne environment
                next_states = env_info.vector_observations         # get next state (for each agent)
                rewards = env_info.rewards                         # get reward (for each agent)
                dones = env_info.local_done                        # see if episode finished
                agent.step(states, actions, rewards, next_states, dones)

                states = next_states

                scores += rewards

                if np.any(dones):
                    break

            # does all the right things with reward tracking
            scores = np.mean(scores)
            mean_reward = reward_tracker.reward(scores, agent.step_t)

            score = np.mean(scores)

            if max_score < score:
                torch.save(agent.actor_local.state_dict(), f'/home/boris/git/udacity/drl/p2_continuous-control/checkpoint_actor_{score:.03f}.pth')
                torch.save(agent.critic_local.state_dict(), f'/home/boris/git/udacity/drl/p2_continuous-control/checkpoint_critic_{score:.03f}.pth')
                max_score = score

                if mean_reward is not None and mean_reward >=  SOLVED_SCORE:
                    solved_episode = i_episode - MEAN_WINDOW - 1
                    print(f"Solved in {solved_episode} episodes")
    env.close()