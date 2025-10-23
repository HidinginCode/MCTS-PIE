"""This module contains the state class which represents the state in a mcts node."""

from __future__ import annotations
from controller import Controller
from agent import Agent

class State():
    """This class contains all metrics needed for evaluation in mcts."""

    def __init__(self, state_controller: Controller):
        """Init method for the state class.

        Args:
            state_controller (Controller): Controller in the state.
        """

        self.state_controller = state_controller.clone()
        self.state_agent = self.state_controller.get_current_agent()
        self.identificator = id(self)

    def clone(self) -> State:
        """Returns a clone of the state without using deepcopy.

        Returns:
            State: Clone of state
        """
        new_state = State.__new__(State)
        new_state.state_controller = self.state_controller.clone()
        new_state.state_agent = new_state.state_controller.get_current_agent()
        new_state.identificator = id(new_state)
        return new_state

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
            "step_count": self.state_agent.get_step_count(),
            "weight_shifted": self.state_agent.get_weight_shifted(),
            "distance_to_goal": self.state_controller.calculate_distance_to_goal()
        }

        return metric_dict

    def get_terminal_state(self) -> bool:
        """Returns if terminal state has been reached.

        Returns:
            bool: Terminal state value
        """

        goal = self.state_controller.map_copy.goal
        current_pos = self.state_controller.current_agent_position
        start = self.state_controller.start_pos
        goal_reached = self.state_controller.goal_collected
        distance = Controller.remaining_roundtrip_distance(current_pos, start, goal, goal_reached)

        return distance == 0
