from model import SharedPolicy
import torch
import numpy as np

from unityagents import UnityEnvironment
from agent import PPOAgent

LR = 1e-03              # learing rate

if __name__ == "__main__":
    env = UnityEnvironment(file_name="Tennis_Win/Tennis")

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

    # create policy to be trained & optimizer
    policy = SharedPolicy(state_size, action_size, seed=33)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    # create agents
    agents = []
    for i in range(1, num_agents + 1):
        agents.append(PPOAgent(i, policy, optimizer))

