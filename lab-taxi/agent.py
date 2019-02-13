import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6, epsilon_start = 1, epsilon_min = 0.00002, gamma = 1.0, alpha= 0.0459):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.actions = np.arange(nA)
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_min = epsilon_min
        self.t = 1

    def get_action_probs(self, state):
        probs = np.repeat(self.epsilon / self.nA, self.nA)
        probs[np.argmax(self.Q[state])] = 1 - np.sum(probs[1:])
        return probs

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        probs = self.get_action_probs(state)
        return np.random.choice(self.nA, p=probs)

    def decay_epsilon(self):
        return 1 / np.power(self.t, 0.53)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        if done:
            next_return = 0
        else:
            next_return = np.dot(self.get_action_probs(next_state), self.Q[next_state])
            #next_return = np.max(self.Q[next_state])

        self.Q[state][action] += self.alpha * (reward + self.gamma * next_return - self.Q[state][action])

        self.epsilon = max(self.decay_epsilon(), self.epsilon_min)
        self.t += 1