from .agent import Agent
import numpy as np


class OptimisticAgent(Agent):

    def __init__(self, bandit, stepper, q_inits, c_bound):
        super().__init__(bandit, stepper, q_inits)
        self._c = c_bound
        self._pick_count = np.zeros(self.arms())

    def argmax(self):
        upper_bounds = self._q_values + self._c * np.sqrt(np.log(self._iter) / (self._pick_count + 1))
        top = float("-inf")
        ties = []

        for idx, q in np.ndenumerate(upper_bounds):
            if q > top:
                top = q
                ties = [idx]
            elif q == top:
                ties.append(idx)
        return np.random.choice(ties)

    def action(self):
        am = self.argmax()
        self._pick_count[am] += 1
        return am

