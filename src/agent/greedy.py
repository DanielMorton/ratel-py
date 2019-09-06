from src.agent.agent import Agent


class GreedyAgent(Agent):

    def __init__(self, bandit, stepper, q_inits):
        super().__init__(bandit, stepper, q_inits)

    def action(self):
        return self._argmax(self._q_star)
