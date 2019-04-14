from model import GaussianPolicyActor, ModelCritic
import torch
import numpy as np
import time
import os

from unityagents import UnityEnvironment
from agent import PPOAgent
import tensorboardX
from utils import RewardTracker, TBMeanTracker
from trajectories import TrajectoryCollector
from collections import defaultdict

LR = 1e-04              # learing rate
LR_CRITIC = 1e-03       # learning rate critic
EPSILON = 0.1           # action clipping param: [1-EPSILON, 1+EPSILON]
BETA = 0.01             # regularization parameter for entropy term
EPOCHS = 4              # train for this number of epochs at a time
TMAX = 3              # maximum trajectory length
MAX_EPISODES = 5000     # episodes
AVG_WIN = 100           # moving average over...
SEED = 1                # leave everything to chance
BATCH_SIZE = 2         # number of tgajectories to collect for learning
SOLVED_SCORE = 0.5      # score at which we are done

GAMMA = 0.99            # discount factor
GAE_LAMBDA = 0.95       # lambda-factor in the advantage estimator for PPO

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    
    env = UnityEnvironment(file_name="p3_collab-compet/Tennis_Linux/Tennis.x86_64")

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]

    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    torch.manual_seed(SEED)

    # create policy to be trained & optimizer
    policy = GaussianPolicyActor(state_size, action_size).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    policy_critic = ModelCritic(state_size).to(device)
    optimizer_critic = torch.optim.Adam(policy_critic.parameters(), lr=LR_CRITIC)

    writer = tensorboardX.SummaryWriter("p3_collab-compet/runs", comment="-mappo")
    
    # create agents
    agents = []
    trajectory_collector = TrajectoryCollector(env, policy, policy_critic, num_agents, tmax=TMAX, gamma=GAMMA, gae_lambda=GAE_LAMBDA)
    tb_tracker = TBMeanTracker(writer, 1)

    for i in range(1, num_agents + 1):
        agents.append(PPOAgent(i, policy, optimizer, policy_critic, optimizer_critic, tb_tracker, EPSILON, BETA))

    n_episodes = 0
    max_score = - np.Inf

    traj_attributes = trajectory_collector.buffer_attrs

    with RewardTracker(writer, mean_window=AVG_WIN) as reward_tracker:

        while True:
            
            start = time.time()
            trajectories = trajectory_collector.create_trajectories()

            n_samples = trajectories[0]['actions'].shape[0]
            n_batches = int((n_samples + 1) / BATCH_SIZE)

            # train agents in a round-robin for the number of epochs
            for epoch in range(EPOCHS):

                for batch in range(n_batches):
                    # convert lists of dictionaries to tensors
                    
                    idx_start = BATCH_SIZE * batch
                    idx_end = idx_start + BATCH_SIZE

                    for i, agent in enumerate(agents):
                        traj_values = defaultdict(list)
                        
                        for k,v in trajectories[i].items():
                            traj_values[k].append(v[idx_start : idx_end])

                        # select the batch of trajectory entries
                        params = [traj_values[k][idx_start : idx_end] for k in traj_attributes]

                        for i in range(len(params)):
                            params[i] = torch.cat(params[i], dim=0)

                        (states, actions, next_states, \
                                                rewards, old_log_probs, dones,
                                                values, advantages, returns) = params

                        # we like all tensors to be shaped (batch_size, value_dims)                                                
                        returns = returns.unsqueeze(1)
                        advantages = advantages.unsqueeze(1)
                                                
                        agent.learn(old_log_probs, states, actions, advantages, returns, values)

            end_time = time.time()

            rewards = trajectory_collector.scores_by_episode[n_episodes : ]

            for idx_r, reward in enumerate(rewards):
                mean_reward = reward_tracker.reward(reward, n_episodes + idx_r, (end_time - start) / 1000.)

                if mean_reward is not None and max_score < mean_reward:
                    if max_score >= SOLVED_SCORE:
                        torch.save(policy.state_dict(), f'p3_collab-compet/checkpoint_actor_{mean_reward:.03f}.pth')
                        torch.save(policy_critic.state_dict(), f'p3_collab-compet/checkpoint_critic_{mean_reward:.03f}.pth')

                    max_score = mean_reward

                    if mean_reward is not None and mean_reward >=  SOLVED_SCORE:
                        solved_episode = n_episodes + idx_r - AVG_WIN - 1
                        print(f"Solved in {solved_episode if solved_episode > 0 else n_episodes + idx_r} episodes")
                        break

            n_episodes += len(rewards)
            if n_episodes >= MAX_EPISODES:
                print(f"Environment not solved")
                break



