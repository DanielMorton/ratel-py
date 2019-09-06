from src.bandit.bandit import Bandit
import numpy as np


class GaussianBandit(Bandit):

    def __init__(self, arms, mean=0, std=1):
        super().__init__(np.random.normal(mean, std, arms))
        self._std = std

    def reward(self, arm):
        return np.random.normal(self._q_values[arm], self._std)
