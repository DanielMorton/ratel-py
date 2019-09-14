import numpy as np

from src.bandit.bandit import Bandit


class GaussianBandit(Bandit):

    def __init__(self, means, stdevs):
        assert means.shape == stdevs.shape
        assert len(means.shape) == 1
        assert stdevs.min() > 0
        super().__init__(means)
        self._std = stdevs

    @property
    def stddevs(self):
        return self._std

    def reward(self, arm):
        return np.random.normal(self.means[arm], self.stddevs[arm])
