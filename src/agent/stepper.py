class Stepper:

    def step(self):
        pass


class ConstantStepper(Stepper):

    def __init__(self, step_size):
        self._step_size = step_size

    def step(self):
        return self._step_size


class HarmonicStepper(Stepper):

    def __init__(self, warmup=1):
        self._step_size = warmup

    def step(self):
        s = 1/self._step_size
        self._step_size += 1
        return s
