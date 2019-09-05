from .agent import Agent
import numpy as np


class OptimisticAgent(Agent):

    def __init__(self, bandit, stepper, q_inits, c_bound):
        super().__init__(bandit, stepper, q_inits)
        self._c = c_bound
        self._pick_count = np.zeros(self.arms())
        assert self.arms == q_inits.shape[0]
        assert len(q_inits.shape) == 1

    def action(self):
        upper_bounds = self._q_values + self._c * np.sqrt(np.log(self._iter) / (self._pick_count + 1))
        am = self._argmax(upper_bounds)
        self._pick_count[am] += 1
        return am
