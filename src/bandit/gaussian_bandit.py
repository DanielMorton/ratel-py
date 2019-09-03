from src.bandit.bandit import Bandit
from numpy.random import normal


class GaussianBandit(Bandit):

    def __init__(self, arms, mean=0, std=1):
        self._q_values = normal(mean, std, arms)
        self._std = std

    def reward(self, arm):
        return normal(self._q_values[arm], 1)
