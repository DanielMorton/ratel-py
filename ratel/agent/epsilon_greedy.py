import numpy as np

from .agent import Agent


class EpsilonGreedyAgent(Agent):
    """Agent that applies the Epsilon-Greedy algorithm to find the optimal bandit strategy. Picks a random arm
    with probablity `epsilon` and the greedy action the rest of the time.

    :param epsilon: Probability of choosing a random action.
    :type epsilon: float
    :param _stepper: List of stepper modules to control update step size for each arm.
    :type _stepper: Stepper
    :param _q_star: Array of estimated values for badit arm means.
    :type _q_star: numpy array
    """

    def __init__(self, stepper, q_inits, epsilon):
        """Constructor Method"""
        super().__init__(stepper, q_inits)
        self._epsilon = epsilon

    def action(self):
        """Action of the agent. Chooses a random arm with probability `epsilon` and the arm with the
        highest estimate the rest of the time. Ties are broken randomly.

        :return: The bandit arm that the agent has chosen to pull.
        :rtype: int
        """
        return np.random.randint(self.arms) if np.random.random() < self._epsilon else np.argmax(self._q_star)

    @property
    def epsilon(self):
        """Returns the probability of picking a random arm.

        :returns: The probability of picking a random arm.
        :rtype: float
        """
        return self._epsilon
