from numpy import ndenumerate
from numpy.random import choice


class Agent:

    def __init__(self, bandit, stepper, q_inits):
        self._bandit = bandit
        self._actions = []
        self._best_action = bandit.best()
        self._stepper = stepper
        self._q_values = q_inits

    def action(self):
        pass

    def agent_step(self):
        current_action = self.action()
        reward = self._reward(current_action)
        self._q_values[current_action] += self._step() * (reward - self._q_values[current_action])

    def argmax(self):
        top = float("-inf")
        ties = []

        for idx, q in ndenumerate(self._q_values):
            if q > top:
                top = q
                ties = [idx]
            elif q == top:
                ties.append(idx)
        return choice(ties)

    def arms(self):
        return self._bandit.arms()

    def _reward(self, current_action):
        return self._bandit.reward(current_action)

    def _step(self):
        return self._stepper.step()