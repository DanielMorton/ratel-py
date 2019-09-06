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

    def step(self):
        return self.step_size


class HarmonicStepper(Stepper):

    def __init__(self, warmup=1):
        self._warmup = warmup
        super().__init__(warmup)

    def reset(self):
        self.__init__(self._warmup)

    def step(self):
        s = 1/self.step_size
        self._step_size += 1
        return s
