from numpy import argmax

class Bandit:

    def arms(self):
        return(self._q_values.shape[0])

    def best(self):
        return argmax(self._q_values)

    def reward(self, arm):
        pass