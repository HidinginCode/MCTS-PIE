"""This module contains the state class which represents the state in a mcts node."""

from copy import deepcopy
from controller import Controller
from agent import Agent

class State():
    """This class contains all metrics needed for evaluation in mcts."""

    def __init__(self, state_controller: Controller):
        """Init method for the state class.

        Args:
            state_controller (Controller): Controller in the state.
        """

        self.state_controller = deepcopy(state_controller)
        self.state_agent = deepcopy(state_controller.get_current_agent())
        self.identificator = id(self)

    def get_state_agent(self) -> Agent:
        """Returns the agent for the current state.

        Returns:
            Agent: Agent in the current state
        """

        return self.state_agent

    def get_state_controller(self) -> Controller:
        """Returns the controller of the current state.

        Returns:
            Controller: Controller of the current state
        """
        return self.state_controller

    def get_identificator(self) -> int:
        """Returns the ID of the current state.

        Returns:
            int: ID of the current state
        """

        return self.identificator

    def get_state_metrics(self) -> dict:
        """Returns a dictionary with the metrics of the current state.

        Returns:
            dict: Dictionary of metrics
        """

        metric_dict = {
            "energy_consumed": self.state_agent.get_energy_consumption(),
            "step_count": self.state_agent.get_step_count(),
            "weight_shifted": self.state_agent.get_weight_shifted(),
            "amount_of_shifts": self.state_agent.get_amount_of_shifts(),
            "position": self.state_controller.get_current_agent_position()
        }

        return metric_dict
