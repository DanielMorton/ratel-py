import numpy as np


class Agent:

    def __init__(self, bandit, stepper, q_inits):
        self._bandit = bandit
        self._actions = []
        self._best_action = bandit.best()
        self._stepper = stepper
        self._q_values = q_inits
        self._iter = 1

    def action(self):
        pass

    def agent_step(self):
        current_action = self.action()
        self._iter += 1
        reward = self._reward(current_action)
        self._q_values[current_action] += self._step() * (reward - self._q_values[current_action])

    @staticmethod
    def _argmax(arr):
        top = float("-inf")
        ties = []

        for idx, q in np.ndenumerate(arr):
            if q > top:
                top = q
                ties = [idx]
            elif q == top:
                ties.append(idx)
        return np.random.choice(ties)

    def arms(self):
        return self._bandit.arms()

    def _reward(self, current_action):
        return self._bandit.reward(current_action)

    def _step(self):
        return self._stepper.step()
