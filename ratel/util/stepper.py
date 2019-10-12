import numpy as np


class Stepper:

    def __init__(self, step_size):
        self._step_size = step_size

    def reset(self):
        pass

    def step(self):
        pass

    @property
    def step_size(self):
        return self._step_size


class ConstantStepper(Stepper):

    def __init__(self, step_size):
        super().__init__(step_size)

    def reset(self):
        self.__init__(self.step_size)

    def step(self, arm=None):
        return self.step_size


class HarmonicStepper(Stepper):

    def __init__(self, warmup=1, length=1):
        self._warmup = warmup * np.ones(length)
        super().__init__(np.copy(self._warmup))

    def reset(self):
        self.__init__(self._warmup[0], self._warmup.shape[0])

    def step(self, arm):
        s = 1/self.step_size[arm]
        self._step_size[arm] += 1
        return s
