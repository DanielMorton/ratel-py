from src.agent.agent import Agent


class GreedyAgent(Agent):

    def __init__(self, stepper, q_inits):
        super().__init__(stepper, q_inits)

    def action(self):
        return self._argmax(self._q_star)
