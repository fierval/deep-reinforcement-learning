import numpy as np

import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TrajectoryCollector:
    """
    Collects trajectories and splits them between agents
    """
    buffer_attrs = [
            "states", "actions", "next_states",
            "rewards", "log_probs", "dones",
            "values", "advantages", "returns"
        ]

    def __init__(self, env, policy, policy_critic, num_agents, tmax=3, gamma = 0.99, gae_lambda = 0.95, debug = False):
        self.env = env
        self.policy = policy
        self.policy_critic = policy_critic
        self.num_agents = num_agents
        self.idx_me = torch.tensor([index+1 for index in range(num_agents)], dtype=torch.float).unsqueeze(1).to(device)

        self.tmax = tmax
        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.debug = debug

        self.rewards = None
        self.scores_by_episode = []
        self.brain_name = None
        self.last_states = None
        self.reset()
        
    @staticmethod
    def to_tensor(x, dtype=np.float32):
        return torch.from_numpy(np.array(x).astype(dtype)).to(device)

    def reset(self):
        self.brain_name = self.env.brain_names[0]
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        self.last_states = self.to_tensor(env_info.vector_observations)

    def calc_returns(self, rewards, values, dones, last_values):
        
        n_step = len(rewards)

        # Create empty buffer
        GAE = torch.zeros(n_step).float().to(device)
        returns = torch.zeros(n_step).float().to(device)

        # Set start values
        GAE_current = torch.zeros(1).float().to(device)
        returns_current = last_values
        values_next = last_values

        for irow in reversed(range(n_step)):
            values_current = values[irow]
            rewards_current = rewards[irow]
            gamma = self.gamma * (1. - dones[irow].float())

            # Calculate TD Error
            td_error = rewards_current + gamma * values_next - values_current
            # Update GAE, returns
            GAE_current = td_error + gamma * self.gae_lambda * GAE_current
            returns_current = rewards_current + gamma * returns_current
            # Set GAE, returns to buffer
            GAE[irow] = GAE_current
            returns[irow] = returns_current

            values_next = values_current

        return GAE, returns

    def create_trajectories(self):
        """
        Inspired by: https://github.com/tnakae/Udacity-DeepRL-p3-collab-compet/blob/master/PPO/agent.py
        Creates trajectories and splites them between all agents, so each one gets individualized trajectories

        Returns:
        A list  of dictionaries, where each list contains a trajectory for its agent
        """

        buffer = []
        
        # tempting to write this as as [{{k: [] for k in self.buffer_attrs}}] * self.num_agents, but it's a bug! :)
        for i in range(self.num_agents):
            buffer.append({k: [] for k in self.buffer_attrs})

        # split trajectory between agents
        for t in range(self.tmax):
            # in order to collect all actions and all rewards we now need to join predicted actions and pipe them 
            # through the environment
            states = self.last_states
            pred = self.policy(states, self.idx_me)
            pred = [v.detach() for v in pred]
            actions, log_probs, _ = pred
            values = self.policy_critic(states).detach()

            # one step forward. We need to move actions to host
            # so we can feed them to the environment
            actions_np = actions.detach().cpu().numpy()

            env_info = self.env.step(actions_np)[self.brain_name]

            for i in range(self.num_agents):
                memory = {}
                memory["states"] = states[i].unsqueeze(0)
                memory["actions"], memory["log_probs"]= actions[i].unsqueeze(0), log_probs[i].unsqueeze(0)
                memory["values"] = values[i].unsqueeze(0)

                memory["next_states"] = self.to_tensor(env_info.vector_observations[i]).unsqueeze(0)
                memory["rewards"] = self.to_tensor(env_info.rewards[i]).unsqueeze(0)
                memory["dones"] = self.to_tensor(env_info.local_done[i], dtype=np.uint8).unsqueeze(0)
                
                # stack one step memory to buffer
                for k, v in memory.items():
                    buffer[i][k].append(v)

            self.last_states = self.to_tensor(env_info.vector_observations)

            r = np.array(env_info.rewards)[None,:]
            if self.rewards is None:
                self.rewards = r
            else:
                self.rewards = np.r_[self.rewards, r]

            if np.array(env_info.local_done).any():
                rewards_mean = self.rewards.sum(axis=0).max()
                self.scores_by_episode.append(rewards_mean)
                self.rewards = None
                self.reset()

        # append remaining rewards
        # TODO: debug only
        if self.debug and self.rewards is not None:
            print("DEBUG: flushing rewards")
            rewards_mean = self.rewards.sum(axis=0).max()
            self.scores_by_episode.append(rewards_mean)
            self.rewards = None
            self.reset()
            
        # append returns and advantages
        values = self.policy_critic(self.last_states).detach()

        for i, b in enumerate(buffer):
            b["advantages"], b["returns"] = self.calc_returns(b["rewards"], b["values"], b["dones"], values[i, :])

            for k, v in b.items():
                if k in ["advantages", "returns"]:
                    continue
                b[k] = torch.cat(v, dim=0)

        return buffer    
