"""This module contains the controller class which facilitates movement and obstacle changes for an agent on the map."""

from __future__ import annotations
from environment import Environment
from directions import Direction

class Controller():
    """Controller class which facilitates map changes and agent movement."""

    def __init__(self, environment: Environment, start_pos: tuple = (0,0)):
        """Init method for the controller class.

        Args:
            environment (Environment): Environment changed by the controller
            start_pos (tuple, optional): Starting position. Defaults to (0,0).
        """
        self._identifier = id(self)
        self._environment = environment.clone()
        self._start_pos = tuple(start_pos)
        self._current_pos = tuple(start_pos)
        self._step_count = 0
        self._weight_shifted = 0.0
        self._distance_to_goal = self.calulate_distance_to_goal()

    def clone(self) -> Controller:
        """Clone method for the controller.

        Returns:
            Controller: Cloned controller
        """
        cloned_env = self._environment.clone()
        clone_controller = Controller(environment=cloned_env, start_pos=self._start_pos)
        clone_controller._current_pos = tuple(self._current_pos)
        clone_controller._step_count = int(self._step_count)
        clone_controller._weight_shifted = float(self._weight_shifted)
        clone_controller._distance_to_goal = float(self._distance_to_goal)
        return clone_controller

    @property
    def identifier(self) -> int:
        """Getter method for identifier.

        Returns:
            int: ID of controller
        """
        return self._identifier

    @property
    def environment(self) -> Environment:
        """Getter method for environment of controller.

        Returns:
            Environment: Environment of controller.
        """
        return self._environment

    @property
    def current_pos(self) -> tuple:
        """Getter method for current_pos.

        Returns:
            tuple: Current position
        """
        return self._current_pos

    @current_pos.setter
    def current_pos(self, position: tuple) -> None:
        """Setter for the current position.

        Args:
            position (tuple): New position on map.
        """
        self._current_pos = position

    @property
    def step_count(self) -> int:
        """Getter method for step count.

        Returns:
            int: Step count
        """
        return self._step_count

    @step_count.setter
    def step_count(self, new_step_count: int) -> None:
        """Step count setter for controller.

        Args:
            new_step_count (int): New step count
        """
        self._step_count = new_step_count

    @property
    def weight_shifted(self) -> float:
        """Getter for weight shifted.

        Returns:
            float: Weight shifted
        """
        return self._weight_shifted

    @weight_shifted.setter
    def weight_shifted(self, new_weight_shifted: float) -> None:
        """Setter for weight shifted.

        Args:
            new_weight_shifted (float): New weight shifted
        """
        self._weight_shifted = new_weight_shifted

    @property
    def distance_to_goal(self) -> float:
        """Getter for distance to goal.

        Returns:
            float: Distance to goal
        """
        return self._distance_to_goal

    @distance_to_goal.setter
    def distance_to_goal(self, new_distance_to_goal: float) -> None:
        """Setter for distance to goal.

        Args:
            new_distance_to_goal (float): New distance to goal
        """
        self._distance_to_goal = new_distance_to_goal

    def move(self, move_dir: Direction, shift_dir: Direction) -> bool:
        """Execute move and obstacle shift on the map.

        Args:
            move_dir (Direction): Direction of the move.
            shift_dir (Direction): Direction of the shift.

        Returns:
            bool: Was the move successfull or not
        """

        # Calculate new position after move
        # This is not necessarily the fanciest method but it is fast
        x, y = self.current_pos
        dx, dy = move_dir.value
        new_pos = (x + dx, y + dy)

        # Calculate shifting position after move
        dx_shift, dy_shift = shift_dir.value
        x_move, y_move = new_pos
        new_shift_pos = (x_move + dx_shift, y_move + dy_shift)

        # If one of both positions is invalid return false
        if not (self.is_valid_positon(new_pos) and self.is_valid_positon(new_shift_pos)):
            return False

        # Move agent to new position, increase steps and calculate new distance to goal
        self._current_pos = new_pos
        self._step_count = self._step_count+1
        self._distance_to_goal = self.calulate_distance_to_goal()

        # Shift weight, increase weight shifted, remove weight from old position
        self._environment._environment[new_shift_pos] += self._environment._environment[new_pos]
        self._weight_shifted += self._environment._environment[new_pos]
        self._environment._environment[new_pos] = 0

    def get_all_valid_pairs(self) -> list[tuple]:
        """Returns a list of all valid movement and shfiting directions pairs at the current position.

        Returns:
            list[tuple]: List of valid pairs.
        """
        valid_pairs = []
        for move_dir in Direction:
            for shift_dir in Direction:
                if self.is_valid_pair((move_dir, shift_dir)):
                    valid_pairs.append((move_dir, shift_dir))
        
        return valid_pairs

    def is_valid_pair(self, pair: tuple[Direction, Direction]) -> bool:
        """Check if a movement and shifiting direction pair is valid.

        Args:
            pair (tuple[Direction, Direction]): Movement and shifting direction pair.

        Returns:
            bool: Valid or not
        """
        move_dir, shift_dir = pair
        x, y = self.current_pos
        dx, dy = move_dir.value
        new_pos = (x + dx, y + dy)

        # Calculate shifting position after move
        dx_shift, dy_shift = shift_dir.value
        x_move, y_move = new_pos
        new_shift_pos = (x_move + dx_shift, y_move + dy_shift)

        if not (self.is_valid_positon(new_pos) and self.is_valid_positon(new_shift_pos)):
            return False
        
        return True

    def is_valid_positon(self, position: tuple) -> bool:
        """Checks if position is inside the map bounds.

        Args:
            position (tuple): Position to be checked

        Returns:
            bool: Is position valid or not
        """

        for coordinate in position:
            if coordinate < 0 or coordinate >= self.environment.env_dim:
                return False
        
        return True

    def calulate_distance_to_goal(self) -> float:
        """Calculates the Manhattan distance to the goal from the current position.

        Returns:
            float: Manhattan distance to goal
        """
        return abs(self._current_pos[0] - self._environment._goal[0]) + abs(self._current_pos[1] - self._environment._goal[1])