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
        stepper = HarmonicStepper()
        assert stepper.step() == 1
        assert stepper.step() == 0.5
        stepper.step()
        assert stepper.step() == 0.25
        stepper.reset()
        assert stepper.step_size == 1

        new_steper = HarmonicStepper(4)
        for _ in range(6):
            new_steper.step()
        assert new_steper.step() == 0.1
        new_steper.reset()
        assert new_steper.step_size == 4
        assert new_steper.step() == 0.25