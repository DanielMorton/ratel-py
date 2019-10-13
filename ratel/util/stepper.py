import numpy as np


class Stepper:
    """Generic class for controlling step sizes for the Agent update algorithms.

    :param _step_size: Array of current update step sizes, one for each arm.
    :type _step_size: numpy array
    """

    def __init__(self, step_size):
        """Constructor method"""
        self._step_size = step_size

    def reset(self):
        """Resets the stepper to its initial conditions"""
        pass

    def step(self):
        """Returns the step size and updates the stepper."""
        pass

    @property
    def step_size(self):
        """Returns the step size.

        :return: The step size.
        :rtype: float
        """
        return self._step_size


class ConstantStepper(Stepper):
    """Stepper with constant step size."""

    def __init__(self, step_size):
        """Constructor method"""
        super().__init__(step_size)

    def reset(self):
        """Resets the stepper to its initial condition."""
        self.__init__(self.step_size)

    def step(self, arm=None):
        """Returns the step size.

        :return: The step size.
        :rtype: float
        """
        return self.step_size


class HarmonicStepper(Stepper):
    """Stepper with step size decreasing in proportion to the harmonic sequence

    :param _warmup: The initial values of the stepper.
    :type _warmup: numpy array
    """

    def __init__(self, warmup=1, length=1):
        """Constructor method"""
        self._warmup = warmup * np.ones(length)
        super().__init__(np.copy(self._warmup))

    def reset(self):
        """Resets the stepper to its initial condition."""
        self.__init__(self._warmup[0], self._warmup.shape[0])

    def step(self, arm):
        """Computes the current step size for the bandit arm and updates stepper for that arm

        :param arm: Bandit arm to be updated.
        :type arm: int
        :return: The current step size for that arm.
        :rtype: float
        """
        s = 1/self.step_size[arm]
        self._step_size[arm] += 1
        return s
