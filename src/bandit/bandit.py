import numpy as np


class Bandit:

    def __init__(self, q_values):
        self._q_values = q_values

    @property
    def arms(self):
        return self._q_values.shape[0]

    @property
    def best(self):
        return np.argmax(self._q_values)

    @property
    def max_reward(self):
        return self._q_values.max()

    def reward(self, arm):
        pass