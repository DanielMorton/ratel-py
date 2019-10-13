import numpy as np

from .bandit import Bandit


class GaussianBandit(Bandit):
    """Bandit with arms that distribute rewards according to Gaussian distributions.
    Each arm has its own mean and variance.

    :param _std: Array of standard deviations for each arm.
    :type _std: numpy array
    """

    def __init__(self, means, stdevs):
        """Constructor method."""
        assert means.shape == stdevs.shape
        assert len(means.shape) == 1
        assert stdevs.min() > 0
        super().__init__(means)
        self._std = stdevs

    @property
    def stddevs(self):
        """Returns the array of bandit arm standard deviations.

        :return: The array of bandit arm standard deviations.
        :rtype: numpy array
        """
        return self._std

    def reward(self, arm):
        """Returns the randomly drawn reward for the arm pulled.

        :param arm: The bandit arm pulled.
        :type arm: int
        :return: The randomly drawn reward for the arm pulled.
        :rtype: float
        """
        return np.random.normal(self.means[arm], self.stddevs[arm])
