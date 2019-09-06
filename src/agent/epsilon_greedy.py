from .agent import Agent
import numpy as np


class EpsilonGreedyAgent(Agent):

    def __init__(self, bandit, stepper, q_inits, epsilon):
        super().__init__(bandit, stepper, q_inits)
        self._epsilon = epsilon

    def action(self):
        return np.random.randint(self.arms) if np.random.random() < self._epsilon else self._argmax(self._q_star)
