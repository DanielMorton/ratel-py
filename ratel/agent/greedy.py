import numpy as np

from .agent import Agent


class GreedyAgent(Agent):
    """Agent that applies the Greedy algorithm to find the optimal bandit strategy. Picks the arm with the highest
    estimated mean value.

    :param _stepper: List of stepper modules to control update step size for each arm.
    :type _stepper: Stepper
    :param _q_star: Array of estimated values for badit arm means.
    :type _q_star: numpy array
    """

    def __init__(self, stepper, q_inits):
        """Constructor Method"""
        super().__init__(stepper, q_inits)

    def action(self):
        """Action of the agent. Chooses the arm with the highest estimated value. Ties are broken randomly

        :return: The bandit arm that the agent has chosen to pull.
        :rtype: int
        """
        return np.argmax(self._q_star)
