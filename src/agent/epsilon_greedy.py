from .agent import Agent
from numpy.random import randint, random


class EpsilonGreedyAgent(Agent):

    def __init__(self, bandit, stepper, q_inits, epsilon):
        super().__init__(bandit, stepper, q_inits)
        assert bandit.arms() == q_inits.shape[0]
        assert len(q_inits.shape) == 1
        self._epsilon = epsilon

    def action(self):
        return randint(self.arms()) if random() < self._epsilon else self.argmax()

