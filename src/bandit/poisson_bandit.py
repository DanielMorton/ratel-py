import numpy as np

from src.bandit.bandit import Bandit


class PoissonBandit(Bandit):

    def __init__(self, means):
        assert means.min() > 0
        super().__init__(means)

    def reward(self, arm):
        return np.random.poisson(self.means[arm])