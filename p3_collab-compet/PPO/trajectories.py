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
            "rewards", "log_probs", "entropy", "dones",
        ]

    def __init__(self, env, policy, num_agents, tmax=300):
        self.env = env
        self.policy = policy
        self.num_agents = num_agents
        self.idx_me = torch.tensor([index+1 for index in range(num_agents)], dtype=torch.float).unsqueeze(0).to(device)
        self.tmax = tmax

        self.rewards = None
        self.scores_by_episode = []
        self.brain_name = None
        self.last_states = [[]] * self.num_agents
        self.reset()
        
    @staticmethod
    def to_tensor(x, dtype=np.float32):
        return torch.from_numpy(np.array(x).astype(dtype)).to(device)

    def reset(self):
        self.brain_name = self.env.brain_names[0]
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        self.last_states = self.to_tensor(env_info.vector_observations)

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
        for _ in range(self.tmax):
            all_memory = [{}] * self.num_agents

            for i in range(self.num_agents):
                memory = all_memory[i]

                # draw action from model
                memory["states"] = self.last_states[i].unsqueeze(0)
                memory["actions"], memory["log_probs"], memory["entropy"], _ = self.policy(memory["states"], self.idx_me[:, i].unsqueeze(0))

            # in order to collect all actions and all rewards we now need to join predicted actions and pipe them 
            # through the environment
            all_actions = torch.cat([memory["actions"] for memory in all_memory], dim=0)

            # one step forward
            actions_np = all_actions.detach().cpu().numpy()

            env_info = self.env.step(actions_np)[self.brain_name]

            for i in range(self.num_agents):
                memory = all_memory[i]

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
                rewards_mean = self.rewards.max(axis=0).mean()
                self.scores_by_episode.append(rewards_mean)
                self.rewards = None
                self.reset()

        for b in buffer:
            for k, v in b.items():
                b[k] = torch.cat(v, dim=0)

        return buffer    
