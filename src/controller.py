"""This module contains the controller for the project.
This controller is used to simulate the agents movements on the map, 
perform the shifts and update the map accordingly."""

from copy import deepcopy
from map import Map
from agent import Agent

class Controller():
    """Controller class, that simulates agent on the map
    and updates map accordingly.
    """

    # Controller Variables
    map_copy = None
    current_agent = None
    identifcator = None

    def __init__(self, map_copy: Map, current_agent: Agent):
        """Init method for the controller class.

        Args:
            map_copy (Map): Map that will be used.
            current_agent (Agent): Agent that is to be simulated.
        """
        self.map_copy = deepcopy(map_copy)
        self.current_agent = current_agent
        self.identifcator = id(self)

    def __str__(self) -> str:
        """String method for the controller.

        Returns:
            str: String that holds `identifier` and `agent identifier`
        """
        return (
            f"Map identifier: {self.identifcator}\n"
            f"Agent identifier: {self.current_agent.get_idenificator()}"
        )

    def get_map_copy(self) -> Map:
        """Returns current map.

        Returns:
            Map: Current map
        """
        return self.map_copy

    def set_map_copy(self, map_copy: Map) -> None:
        """Sets a map for the controller.

        Args:
            map_copy (Map): Map to be set
        """
        self.map_copy = deepcopy(map_copy)

    def get_current_agent(self) -> Agent:
        """Returns current agent used by controller.

        Returns:
            Agent: Current agent
        """
        return self.current_agent

    def set_current_agent(self, current_agent: Agent) -> None:
        """Sets the current agent for the controller.

        Args:
            current_agent (Agent): Agent to be set for controller
        """
        self.current_agent = current_agent

    def get_identificator(self) -> int:
        """Returns identificator of controller.

        Returns:
            int: Identifier
        """
        return self.identifcator
