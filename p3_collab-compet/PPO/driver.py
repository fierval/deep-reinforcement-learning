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
    trajectory_collector = TrajectoryCollector(env, policy, num_agents, tmax=TMAX)
    tb_tracker = TBMeanTracker(writer, 1)

    for i in range(1, num_agents + 1):
        agents.append(PPOAgent(i, policy, optimizer, policy_critic, optimizer_critic, tb_tracker, EPSILON, BETA))

    n_episodes = 0
    max_score = - np.Inf

    traj_attributes = ["log_probs", "entropy", "states", "actions", "rewards", "dones"]
    with RewardTracker(writer, mean_window=AVG_WIN) as reward_tracker:

        while True:
            
            trajectories = []
            for i in range(num_agents):
                trajectories.append([])
                
            start = time.time()
            for j in range(BATCH_SIZE):
                trajectories_step = trajectory_collector.create_trajectories()

                for i in range(num_agents):
                    trajectories[i].append(trajectories_step[i])

            for epoch in range(EPOCHS):
                for i, agent in enumerate(agents):
                    old_log_probs, entropies, states, actions, rewards, dones = [], [], [], [], [], []
                    traj_values = old_log_probs, entropies, states, actions, rewards, dones
                    
                    # convert lists of dictionaries to tensors
                    for t in trajectories[i]:
                        for k, t_attr in enumerate(traj_attributes):
                            traj_values[k].append(t[t_attr])
                    for t_value in traj_values:
                        t_value = torch.cat(t_value, dim=0)

                    agent.learn(old_log_probs, entropies, states, actions, rewards, dones)

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



