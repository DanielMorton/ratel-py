import numpy as np

from src.bandit.bandit import Bandit


class BernoulliBandit(Bandit):

    def __init__(self, probs):
        assert probs.max() <= 1
        assert probs.min() >= 0
        super().__init__(probs)

    @@property
    def probs(self):
        return self._q_values

    def reward(self, arm):
        return np.random.binomial(1, self.probs[arm], 1)