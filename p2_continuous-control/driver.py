import numpy as np
import random
import time
import copy, math
from collections import namedtuple, deque

from utils import distr_projection, RewardTracker, TBMeanTracker

import torch
import torch.nn.functional as F
import torch.optim as optim
import threading

# use tensorboard to monitor progress
from d4pg_agent import D4PGAgent
from ddpg_agent import Agent, N_EPISODES, MAX_T

if __name__ == '__main__':
    from unityagents import UnityEnvironment  

    SOLVED_SCORE = 30
    MEAN_WINDOW = 100

    env = UnityEnvironment(file_name='p2_continuous-control/Reacher_Windows_20/Reacher')

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

    #agent = D4PGAgent(num_agents, state_size, action_size, 1)
    agent = Agent(num_agents, state_size, action_size, 1)
    max_score = -np.Inf
    solved_episode = -np.Inf

    # tracks all the mean rewards etc
    with RewardTracker(agent.writer, MEAN_WINDOW) as reward_tracker:

        for i_episode in range(1, n_episodes+1):
            env_info = env.reset(train_mode=True)[brain_name]
            states = env_info.vector_observations
            scores = np.zeros(num_agents)
            
            # noise reset
            #agent.reset()
            start_time = time.time()
            for t in range(max_t):
                actions = agent.act(states)

                env_info = env.step(actions)[brain_name]           # send all actions to tne environment
                next_states = env_info.vector_observations         # get next state (for each agent)
                rewards = env_info.rewards                         # get reward (for each agent)
                dones = env_info.local_done                        # see if episode finished
                for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                    agent.step(state, action, reward, next_state, done, t)             
                states = next_states

                scores += rewards

                if np.any(dones):
                    break

            # does all the right things with reward tracking
            duration = time.time() - start_time
            score = np.mean(scores)
            mean_reward = reward_tracker.reward(scores, score, agent.step_t, duration)

            if max_score < score:
                if max_score >= SOLVED_SCORE:
                    torch.save(agent.actor_local.state_dict(), f'p2_continuous-control/checkpoint_actor_{score:.03f}.pth')
                    torch.save(agent.critic_local.state_dict(), f'p2_continuous-control/checkpoint_critic_{score:.03f}.pth')

                max_score = score

                if mean_reward is not None and mean_reward >=  SOLVED_SCORE:
                    solved_episode = i_episode - MEAN_WINDOW - 1
                    print(f"Solved in {solved_episode} episodes")

                    break
    env.close()