import numpy as np

from .bandit import Bandit


class PoissonBandit(Bandit):
    """Bandit with arms that distribute rewards according to a Poisson distribution. Each arm has its own mean."""

    def __init__(self, means):
        """Constructor method."""
        assert means.min() > 0
        super().__init__(means)

    def reward(self, arm):
        """Returns the randomly drawn reward for the arm pulled.

        :param arm: The bandit arm pulled.
        :type arm: int
        :return: The randomly drawn reward for the arm pulled.
        :rtype: float
        """
        return np.random.poisson(self.means[arm])
