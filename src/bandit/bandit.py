import numpy as np


class Bandit:

    def arms(self):
        return self._q_values.shape[0]

    def best(self):
        return np.argmax(self._q_values)

    def reward(self, arm):
        pass