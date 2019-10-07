import numpy as np


class Agent:
    """Generic agent for playing multi-arm bandit games."""

    def __init__(self, stepper, q_inits):
        assert len(q_inits.shape) == 1
        assert q_inits.shape[0] > 0
        self._stepper = stepper
        self._q_star = q_inits

    @staticmethod
    def _argmax(arr):
        top = float("-inf")
        ties = []

        for idx, q in np.ndenumerate(arr):
            if q > top:
                top = q
                ties = [idx[0]]
            elif q == top:
                ties.append(idx[0])
        return np.random.choice(ties)

    def _step(self, arm):
        return self._stepper.step(arm)

    def action(self):
        pass

    @property
    def arms(self):
        return self._q_star.shape[0]

    def current_estimate(self, arm):
        return self._q_star[arm]

    def reset(self, q_inits):
        self._stepper.reset()
        self._q_star = q_inits

    def step(self, current_action, reward):
        self._q_star[current_action] += self._step(current_action) * (reward - self._q_star[current_action])
