import numpy as np


class Bandit:
    """Base class for multi-armed bandits. Contains all the methods common to all bandit classes.

    :param _q_values: Array of mean values for each bandit arm.
    :type _q_values: numpy array
    """

    def __init__(self, q_values):
        """Constructor method."""
        assert len(q_values.shape) == 1
        assert q_values.shape[0] > 0
        self._q_values = q_values

    @property
    def arms(self):
        """Returns the number of bandit arms.

        :return: The number of bandit arms.
        :rtype: int
        """
        return self._q_values.shape[0]

    @property
    def best_arm(self):
        """Returns the bandit arm with the highest average value. In case of ties, the first arm is chosen.

        :return: The bandit arm with the highest average value.
        :rtype: int
        """
        return np.argmax(self._q_values)

    @property
    def max_reward(self):
        """"""
        return self._q_values.max()

    @property
    def means(self):
        return self._q_values

    def reward(self, arm):
        pass
