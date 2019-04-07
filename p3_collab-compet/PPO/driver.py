from model import GaussianPolicyActor, ModelCritic
import torch
import numpy as np

from unityagents import UnityEnvironment
from agent import PPOAgent
import tensorboardX
from utils import RewardTracker
from trajectories import TrajectoryCollector

LR = 1e-04              # learing rate
LR_CRITIC = 1e-03       # learning rate critic
EPSILON = 0.1           # action clipping param: [1-EPSILON, 1+EPSILON]
BETA = 0.01             # regularization parameter for entropy term
EPOCHS = 4              # train for this number of epochs at a time
TMAX = 300              # maximum trajectory length
MAX_EPISODES = 5000     # episodes
AVG_WIN = 100           # moving average over...
SEED = 1                # leave everything to chance

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
    torch.manual_seed(SEED)

    # create policy to be trained & optimizer
    policy = GaussianPolicyActor(state_size, action_size)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    policy_critic = ModelCritic(state_size)
    optimizer_critic = torch.optim.Adam(policy_critic.parameters(), lr=LR_CRITIC)

    writer = tensorboardX.SummaryWriter(comment="-mappo")
    
    # create agents
    agents = []
    trajectories = TrajectoryCollector(env, policy, num_agents, tmax=TMAX)

    for i in range(1, num_agents + 1):
        agents.append(PPOAgent(i, policy, optimizer, policy_critic, optimizer_critic, EPOCHS, EPSILON, BETA))

    with RewardTracker(writer, mean_window=AVG_WIN) as reward_tracker:
        for episode in range(MAX_EPISODES):

            env_info = env.reset(train_mode=True)[brain_name]
            states = env_info.vector_observations

            for t in range(TMAX):
                actions = [agent.act(state).squeeze().cpu().detach().numpy() for state, agent in zip(states, agents)]
                actions = np.vstack(actions)

                env_info = env.step(actions)[brain_name]

