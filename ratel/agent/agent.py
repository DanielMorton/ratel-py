import numpy as np


class Agent:
    """Generic agent for playing multi-arm bandit games.

    :param _stepper: List of stepper modules to control update step size for each arm.
    :type _stepper: Stepper
    :param _q_star: Array of estimated values for badit arm means.
    :type _q_star: numpy array
    """

    def __init__(self, stepper, q_inits):
        """Constructor Method"""
        assert len(q_inits.shape) == 1
        assert q_inits.shape[0] > 0
        self._stepper = stepper
        self._q_star = q_inits

    @staticmethod
    def _argmax(arr):
        """Returns the argmax of a numpy array. Ties broken randomly.

        :param arr (numpy array): Array of numbers.
        :rtype: int
        """
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
        """Returns the current step size for the given arm.

        :param arm: Arm for which step size is returned.
        :type arm: int
        :return: The current step size for the given arm.
        :rtype: float
        """
        return self._stepper.step(arm)

    def action(self):
        """Action of the agent.

        :return: The bandit the arm that the agent has chosen to pull.
        :rtype: int
        """
        pass

    @property
    def arms(self):
        """Returns the number of arms on the bandit.

        :return: The number of arms on the bandit.
        :rtype: int
        """
        return self._q_star.shape[0]

    def current_estimate(self, arm):
        """Returns the current estimate for the value of the given arm.

        :param arm: Arm for which step size is returned.
        :type arm: int

        :return: The current estimate for the value of the given arm.
        :rtype: float
        """
        return self._q_star[arm]

    def reset(self, q_inits):
        """Resets the agent with a new set of initial values.

        :param q_inits: a new set of initial values for q_star.
        :type q_inits: numpy array
        """
        self._stepper.reset()
        self._q_star = q_inits

    def step(self, current_action, reward):
        """Updates q_start for the current action based on the most recent reward.

        :param current_action: The bandit arm selected.
        :type current_action: int
        :param reward: The return generated from the most recent bandit arm.
        :type reward: float
        """
        self._q_star[current_action] += self._step(current_action) * (reward - self._q_star[current_action])
