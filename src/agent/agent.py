import numpy as np


class Agent:

    def __init__(self, bandit, stepper, counter, q_inits):
        self._bandit = bandit
        self._actions = []
        self._best_action = bandit.best
        self._stepper = stepper
        self._counter = counter
        self._q_values = q_inits
        self._tot_reward = 0

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

    def _reward(self, current_action):
        return self._bandit.reward(current_action)

    def _step(self):
        return self._stepper.step()

    def action(self):
        pass

    def agent_step(self):
        current_action = self.action()
        self._actions.append(current_action)
        reward = self._reward(current_action)
        self._counter.iterate(reward)
        self._tot_reward += reward
        self._q_values[current_action] += self._step() * (reward - self._q_values[current_action])

    @property
    def arms(self):
        return self._bandit.arms

    @property
    def max_reward(self):
        return self._bandit.max_reward
