from unittest import TestCase

import numpy as np

from ratel.bandit.gaussian_bandit import GaussianBandit


class TestGaussianBandit(TestCase):
    def test_gaussian_bandit(self):
        bandit = GaussianBandit(np.array([-1.35, -0.71, 0.13, 0.06, -0.28, 0.05, 0.28, 0.49, 0.38, 1.32]),
                                np.array([0.68, 1.99, 0.16, 0.47, 1.77, 0.15, 1.05, 1.48, 0.03, 0.87]))
        assert np.array_equal(bandit.means, np.array([-1.35, -0.71, 0.13, 0.06, -0.28, 0.05, 0.28, 0.49, 0.38, 1.32]))
        assert np.array_equal(bandit.stddevs, np.array([0.68, 1.99, 0.16, 0.47, 1.77, 0.15, 1.05, 1.48, 0.03, 0.87]))
        assert bandit.arms == 10
        assert bandit.best_arm == 9
        assert bandit.max_reward == 1.32

    def test_mismatched_bandit(self):
        self.assertRaises(AssertionError,
                          lambda: GaussianBandit(np.array([-1.35, -0.71, 0.13, 0.06, -0.28, 0.05, 0.28, 0.49, 0.38, 1.32]),
                                                 np.array([0.68, 1.99, 0.16, 0.47, 1.77, 0.15, 1.05, 1.48])))

    def test_multidim_bandit(self):
        self.assertRaises(AssertionError,
                          lambda: GaussianBandit(np.array([[-1.35, -0.71, 0.13, 0.06, -0.28], [0.05, 0.28, 0.49, 0.38, 1.32]]),
                                         np.array([[0.68, 1.99, 0.16, 0.47, 1.77], [0.15, 1.05, 1.48, 0.03, 0.87]])))

    def test_bad_input(self):
        self.assertRaises(AssertionError,
                          lambda : GaussianBandit(np.array([-1.35, -0.71, 0.13, 0.06, -0.28, 0.05, 0.28, 0.49, 0.38, 1.32]),
                                np.array([-1.35, -0.71, 0.13, 0.06, -0.28, 0.05, 0.28, 0.49, 0.38, 1.32])))
