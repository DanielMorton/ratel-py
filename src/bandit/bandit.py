import numpy as np


class Bandit:

    def __init__(self, q_values):
        assert len(q_values.shape) == 1
        assert q_values.shape[0] > 0
        self._q_values = q_values

    @property
    def arms(self):
        return self._q_values.shape[0]

    @property
    def best_arm(self):
        return np.argmax(self._q_values)

    @property
    def max_reward(self):
        return self._q_values.max()

    @property
    def means(self):
        return self._q_values

    def reward(self, arm):
        pass