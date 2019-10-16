import numpy as np

from .agent import Agent


class OptimisticAgent(Agent):
    """Agent that applies the Optimistic algorithm to find the optimal bandit strategy. Picks the arm with the highest
        upper confidence bound.

        :param _stepper: List of stepper modules to control update step size for each arm.
        :type _stepper: Stepper
        :param _q_star: Array of estimated values for badit arm means.
        :type _q_star: numpy array
        :param _c: Size of the confidence range.
        :type _c: float
        """

    def __init__(self, stepper, q_inits, c_bound):
        """Constructor method."""
        super().__init__(stepper, q_inits)
        self._c = c_bound
        self._pick_count = np.zeros(self.arms())

    def action(self):
        """Action of the agent. Chooses the arm with the highest upper confidence bound. Ties are broken randomly

        :return: The bandit arm that the agent has chosen to pull.
        :rtype: int
        """
        upper_bounds = self._q_star + self._c * np.sqrt(np.log(self.counter) / (self._pick_count + 1))
        am = np.argmax(upper_bounds)
        self._pick_count[am] += 1
        return am

    @property
    def c(self):
        """Returns the size of the confidence range.

        :return: The size of the confidence range.
        :rtype: float
        """
        return self._c
