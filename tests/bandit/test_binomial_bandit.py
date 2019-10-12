from unittest import TestCase

import numpy as np

from ratel.bandit.binomial_bandit import BinomialBandit, BernoulliBandit


class TestBinomialBanidt(TestCase):
    def test_binomial_bandit(self):
        bandit = BinomialBandit(np.array([0.89, 0.5, 0.13, 0.88, 0.8, 0.2, 0.15, 0.79, 0.81, 0.01]),
                                np.array([3, 4, 12, 14, 7, 15, 6, 20, 16, 11]))
        assert np.array_equal(bandit.probs, np.array([0.89, 0.5, 0.13, 0.88, 0.8, 0.2, 0.15, 0.79, 0.81, 0.01]))
        assert np.array_equal(bandit.nums, np.array([3, 4, 12, 14, 7, 15, 6, 20, 16, 11]))
        assert np.allclose(bandit.means, np.array([2.67, 2.0, 1.56, 12.32, 5.6, 3.0, 0.9, 15.8, 12.96, 0.11]))
        assert bandit.arms == 10
        assert bandit.best_arm == 7
        assert bandit.max_reward == 15.8

    def test_mismatched_bandit(self):
        self.assertRaises(AssertionError,
                          lambda: BinomialBandit(
                              np.array([0.89, 0.5, 0.13, 0.88, 0.8, 0.2, 0.15, 0.79, 0.81, 0.01]),
                              np.array([3, 4, 12, 14, 7, 15, 6, 20])))

    def test_multidim_bandit(self):
        self.assertRaises(AssertionError,
                          lambda: BinomialBandit(
                              np.array([[0.89, 0.5, 0.13, 0.88, 0.8], [0.2, 0.15, 0.79, 0.81, 0.01]]),
                              np.array([[3, 4, 12, 14, 7], [15, 6, 20, 16, 11]])))

    def test_bad_input(self):
        self.assertRaises(AssertionError,
                          lambda: BinomialBandit(
                              np.array([3, 4, 12, 14, 7, 15, 6, 20]),
                              np.array([3, 4, 12, 14, 7, 15, 6, 20])))
        self.assertRaises(AssertionError,
                          lambda: BinomialBandit(np.array([-0.89, 0.5, 0.13, 0.88, 0.8, 0.2, 0.15, 0.79, 0.81, 0.01]),
                                                 np.array([3, 4, 12, 14, 7, 15, 6, 20, 16, 11])))
        self.assertRaises(AssertionError,
                          lambda: BinomialBandit(np.array([0.89, 0.5, 0.13, 0.88, 0.8, 0.2, 0.15, 0.79, 0.81, 0.01]),
                                                 np.array([-3, 4, 12, 14, 7, 15, 6, 20, 16, 11])))


class TestBernoulliBandit(TestCase):
    def test_bernoulliBandit(self):
        bandit = BernoulliBandit(np.array([0.89, 0.5, 0.13, 0.88, 0.8, 0.2, 0.15, 0.79, 0.81, 0.01]))
        assert np.array_equal(bandit.probs, np.array([0.89, 0.5, 0.13, 0.88, 0.8, 0.2, 0.15, 0.79, 0.81, 0.01]))
        assert np.array_equal(bandit.means, np.array([0.89, 0.5, 0.13, 0.88, 0.8, 0.2, 0.15, 0.79, 0.81, 0.01]))
        assert np.array_equal(bandit.nums, np.ones(10))
        assert bandit.arms == 10
        assert bandit.best_arm == 0
        assert bandit.max_reward == 0.89

    def test_bad_input(self):
        self.assertRaises(AssertionError,
                          lambda: BernoulliBandit(np.array([1.89, 0.5, 0.13, 0.88, 0.8, 0.2, 0.15, 0.79, 0.81, 0.01])))
        self.assertRaises(AssertionError,
                          lambda: BernoulliBandit(np.array([-0.89, 0.5, 0.13, 0.88, 0.8, 0.2, 0.15, 0.79, 0.81, 0.01])))
