import numpy as np

from .agent import Agent


class EpsilonGreedyAgent(Agent):

    def __init__(self, stepper, q_inits, epsilon):
        super().__init__(stepper, q_inits)
        self._epsilon = epsilon

    def action(self):
        return np.random.randint(self.arms) if np.random.random() < self._epsilon else self._argmax(self._q_star)
