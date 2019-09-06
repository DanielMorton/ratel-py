import numpy as np
from src.bandit.bandit import Bandit


class TestBandit:
    def test_bandit(self):
        bandit = Bandit(np.array([0.2, 1.0, 0.4, 0.8, 0.6]))
        assert bandit.arms == 5
        assert bandit.best_arm == 1
        assert bandit.max_reward == 1.0
