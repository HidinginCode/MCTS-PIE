"""This module contains the controller for the project.
This controller is used to simulate the agents movements on the map, 
perform the shifts and update the map accordingly."""

from __future__ import annotations
from map import Map
from agent import Agent
from directions import Direction

class Controller():
    """Controller class, that simulates agent on the map
    and updates map accordingly.
    """

    def __init__(self, map_copy: Map, current_agent: Agent, start_pos: tuple = (0,0)):
        """Init method for the controller class.

        Args:
            map_copy (Map): Map that will be used.
            current_agent (Agent): Agent that is to be simulated.
        """
        self.map_copy = map_copy.clone()
        self.current_agent = current_agent.clone()
        self.identificator = id(self)
        self.current_agent_position = start_pos

    def __str__(self) -> str:
        """String method for the controller.

        Returns:
            str: String that holds `identifier` and `agent identifier`
        """
        return (
            f"Map identifier: {self.identificator}\n"
            f"Agent identifier: {self.current_agent.get_idenificator()}"
            f"Current agent position: {self.current_agent_position}"
        )

    def clone(self) -> Controller:
        """Clones a controller without using deepcopy.

        Returns:
            Controller: Cloned controller
        """
        new_controller = Controller.__new__(Controller)
        new_controller.map_copy = self.map_copy.clone()
        new_controller.current_agent = self.current_agent.clone()
        new_controller.current_agent_position = tuple(self.current_agent_position)
        new_controller.identificator = id(new_controller)
        return new_controller

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
        self.map_copy = map_copy.clone()

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
        self.current_agent = current_agent.clone()

    def get_identificator(self) -> int:
        """Returns identificator of controller.

        Returns:
            int: Identifier
        """
        return self.identificator

    def get_current_agent_position(self) -> tuple:
        """Returns current position of the agent.

        Returns:
            tuple: Agent coordinates (x,y)
        """
        return self.current_agent_position

    def set_current_agent_position(self, new_pos: tuple) -> None:
        """Sets current position of the agent.

        Args:
            new_pos (tuple): New position of the agent (x,y)
        """
        self.current_agent_position = new_pos

    def is_valid_position(self, pos:tuple) -> bool:
        """Checks if a position is inside of the map boundaries.

        Args:
            pos (tuple): Position to be checked (x,y).

        Returns:
            bool: Truth value for position beeing in map bounds
        """

        for coordinate in pos:
            if coordinate < 0 or coordinate >= self.map_copy.get_map_dim():
                return False

        return True

    def is_valid_direction(self, direction: Direction, position: tuple | None = None) -> bool:
        """Checks if a direction is valid from a given position.
        This check is based on the agents current position if no position is specified.

        Args:
            direction (Direction): Direction to move in

        Returns:
            bool: Valid state
        """

        base_position = position or self.current_agent_position
        new_pos = tuple(sum(coord) for coord in zip(base_position, direction.value))
        return self.is_valid_position(new_pos)

    def combine_obstacles(self, obstacle_pos: tuple, destination: tuple) -> None:
        """Combines the densities of 2 obstacles at specified positions.

        Args:
            obstacle_pos (tuple): Position of the obstacle that has to be shifted (x,y).
            destination (tuple): Position of destination obtacle (x,y).
        """

        obstacle_density = self.map_copy.get_obstacle_density(pos = obstacle_pos)
        self.map_copy.change_obstacle_density(pos = destination, density = obstacle_density)
        self.map_copy.eliminate_obstacle(pos = obstacle_pos)

    def calculate_distance_to_goal(self) -> float:
        """Calculates the agents distance to the goal.

        Returns:
            float: Distance to goal
        """

        goal = self.map_copy.get_goal()
        position = self.current_agent_position

        return sum(abs(a - b) for a, b in zip(position, goal))

    def move_agent(self, move_direction: Direction, shifting_direction: Direction) -> bool:
        """Moves the agent and facilitates map changes.

        Args:
            new_pos (Direction): Position to move the agent to (x,y).
            shifting_direction (Direction): Direction the obstacle is shifted towards.

        Returns:
            bool: Check if move was sucessfull.
        """

        new_pos = tuple(
            sum(coord) for coord in zip(
                self.current_agent_position,
                move_direction.value
            )
        )

        # Check if new position is valid
        if not self.is_valid_position(new_pos):
            return False

        # Check for obstacle on new_pos
        # If there is one we need to do the shifting action
        if self.map_copy.get_obstacle_density(new_pos) > 0:
            new_obstacle_pos = tuple(sum(coord) for coord in zip(new_pos, shifting_direction.value))

            if not self.is_valid_position(new_obstacle_pos):
                return False

            obstacle_density_on_pos = self.map_copy.get_obstacle_density(new_pos)
            self.current_agent.increase_weight_shifted(obstacle_density_on_pos)
            self.combine_obstacles(new_pos, new_obstacle_pos)
            self.current_agent.increase_amount_of_shifts()

        self.set_current_agent_position(new_pos=new_pos)
        self.current_agent.increase_step_count()

        if new_pos == self.map_copy.get_goal():
            self.current_agent.set_goal_collected(True)

        return True
