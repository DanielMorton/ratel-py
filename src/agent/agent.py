import numpy as np
from src.util.counter import RecordCounter


class Agent:

    def __init__(self, bandit, stepper, q_inits):
        assert len(q_inits.shape) == 1
        assert q_inits.shape[0] > 0
        self._bandit = bandit
        assert self.arms == q_inits.shape[0]
        self._stepper = stepper
        self._counter = RecordCounter()
        self._wins = RecordCounter()
        self._q_star = q_inits

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
        self._wins.iterate(current_action == self.best_arm)
        reward = self._reward(current_action)
        self._counter.iterate(reward)
        self._q_star[current_action] += self._step() * (reward - self._q_star[current_action])

    @property
    def arms(self):
        return self._bandit.arms

    @property
    def best_arm(self):
        return self._bandit.best_arm

    @property
    def counter(self):
        return self._counter.counter

    @property
    def max_reward(self):
        return self._bandit.max_reward

    def reset(self, q_inits):
        self._stepper.reset()
        self._counter.reset()
        self._wins.reset()
        self._q_star = q_inits

    def run(self, steps):
        for _ in range(steps):
            self.agent_step()

