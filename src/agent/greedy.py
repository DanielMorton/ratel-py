from src.agent.agent import Agent


class GreedyAgent(Agent):

    def __init__(self, bandit, stepper, q_inits):
        super().__init__(bandit, stepper, q_inits)
        assert self.arms == q_inits.shape[0]
        assert len(q_inits.shape) == 1

    def action(self):
        return self._argmax(self._q_values)