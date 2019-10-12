import pandas as pd

from .util import RecordCounter


class Optimizer:

    def __init__(self, agent, bandit):
        assert agent.arms == bandit.arms
        self._agent = agent
        self._bandit = bandit
        self._rewards = RecordCounter()
        self._wins = RecordCounter()

    def _reward(self, current_action):
        return self._bandit.reward(current_action)

    @property
    def counter(self):
        return self._rewards.counter

    def output_df(self):
        return pd.DataFrame({'wins': self._wins.record, 'rewards': self._rewards.record})

    def pull_arm(self):
        current_action = self._agent.action()
        self._wins.iterate(current_action == self._bandit.best_arm)
        reward = self._reward(current_action)
        self._rewards.iterate(reward)
        self._agent.step(current_action, reward)

    def reset(self, q_inits):
        self._agent.reset(q_inits)
        self._rewards.reset()
        self._wins.reset()

    def run(self, steps):
        for _ in range(steps):
            self.pull_arm()
        return self.output_df()


