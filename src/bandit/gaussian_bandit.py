import numpy as np

from src.bandit.bandit import Bandit


class GaussianBandit(Bandit):

    def __init__(self, means, stdevs):
        super().__init__(means)
        self._std = stdevs

    def reward(self, arm):
        return np.random.normal(self._q_values[arm], self._std[arm])
