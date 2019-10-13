import numpy as np

from .bandit import Bandit


class BinomialBandit(Bandit):
    """Bandit with arms that distribute rewards according to binomial distributions. Each arm has a distribution
    equivalent to doing a fixed number of Bernoulli trials with a fixed probability. Each arm can be a different number
    of trials and a different probability of success. Mean values for each arm are determined by multiplying the number
    of trials by the probability of success.

    :param _probs: Array of probabilities, one for each arm.
    :type _probs: numpy array
    :param _nums: Array of integer number of trials, one for each arm.
    :type _nums: numpy array
    """

    def __init__(self, probs, nums):
        """Constructor method."""
        assert probs.max() <= 1
        assert probs.min() >= 0
        assert nums.min() > 0
        assert probs.shape == nums.shape
        super().__init__(probs * nums)
        self._probs = probs
        self._nums = nums

    @property
    def nums(self):
        """Returns the array of trial numbers for each Binomial distribution.

        :return: The array of trial numbers for each Binomial distribution.
        :rtype: numpy array
        """
        return self._nums

    @property
    def probs(self):
        """Returns the array of trial probabilities for each Binomial distribution.

        :return: The array of trial probabilities for each Binomial distribution.
        :rtype: numpy array
        """
        return self._probs

    def reward(self, arm):
        """Returns the randomly drawn reward for the arm pulled.

        :param arm: The bandit arm pulled.
        :type arm: int
        :return: The randomly drawn reward for the arm pulled.
        :rtype: int
        """
        return np.random.binomial(self.nums[arm], self.probs[arm])


class BernoulliBandit(BinomialBandit):
    """A special case of the Binomial Bandit where the number of trials is always 1."""

    def __init__(self, probs):
        """Constructor method."""
        super().__init__(probs, np.ones(probs.shape))
