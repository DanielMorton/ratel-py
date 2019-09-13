from unittest import TestCase

import numpy as np

from src.bandit.bandit import Bandit


class TestBandit(TestCase):
    def test_bandit(self):
        bandit = Bandit(np.array([0.86, 0.46, 0.95, 0.58, 0.24]))
        assert bandit.arms == 5
        assert bandit.best_arm == 2
        assert bandit.max_reward == 0.95
        assert np.array_equal(bandit.means, np.array([0.86, 0.46, 0.95, 0.58, 0.24]))

    def test_other_bandit(self):
        bandit = Bandit(np.array([0.18, 0.91, 0.41, 0.93, 0.61, 0.43, 0.35, 0.9, 0.67]))
        assert bandit.arms == 9
        assert bandit.best_arm == 3
        assert bandit.max_reward == 0.93
        assert np.array_equal(bandit.means, np.array([0.18, 0.91, 0.41, 0.93, 0.61, 0.43, 0.35, 0.9, 0.67]))

    def test_empty_bandit(self):
        self.assertRaises(AssertionError, Bandit, np.array([]))

    def test_multidim_bandit(self):
        self.assertRaises(AssertionError, Bandit, np.array([[0.1, 0.8], [0.5, 0.1]]))
