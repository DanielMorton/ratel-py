import numpy as np

from .bandit import Bandit


class BinomialBandit(Bandit):

    def __init__(self, probs, nums):
        assert probs.max() <= 1
        assert probs.min() >= 0
        assert nums.min() > 0
        assert probs.shape == nums.shape
        super().__init__(probs * nums)
        self._probs = probs
        self._nums = nums

    @property
    def nums(self):
        return self._nums

    @property
    def probs(self):
        return self._probs

    def reward(self, arm):
        return np.random.binomial(self.nums[arm], self.probs[arm])


class BernoulliBandit(BinomialBandit):

    def __init__(self, probs):
        super().__init__(probs, np.ones(probs.shape))
