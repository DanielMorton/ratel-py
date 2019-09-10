import numpy as np

from src.util.stepper import ConstantStepper, HarmonicStepper, Stepper


class TestStepper:
    def test_stepper(self):
        stepper = Stepper(0.1)
        assert stepper.step_size == 0.1

    def test_constant_stepper(self):
        stepper = ConstantStepper(0.1)
        assert stepper.step() == 0.1
        stepper.reset()
        assert stepper.step() == 0.1

    def test_harmonic_stepper(self):
        stepper = HarmonicStepper(length=1)
        assert stepper.step(0) == 1
        assert stepper.step(0) == 0.5
        stepper.step(0)
        assert stepper.step(0) == 0.25
        stepper.reset()
        assert np.array_equal(stepper.step_size, np.array([1]))

        new_steper = HarmonicStepper(warmup=4)
        for _ in range(6):
            new_steper.step(0)
        assert new_steper.step(0) == 0.1
        new_steper.reset()
        assert np.array_equal(new_steper.step_size, np.array([4]))
        assert new_steper.step(0) == 0.25