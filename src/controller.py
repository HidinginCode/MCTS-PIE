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
        self._goal_collected = False
        self._distance_to_goal = self.calculate_distance_to_goal()

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
        clone_controller._goal_collected = bool(self._goal_collected)
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
            print("INVALID MOVE " * 100)
            return False

        # Move agent to new position, increase steps and calculate new distance to goal
        self._current_pos = new_pos
        self._step_count = self._step_count+1
        if self._current_pos == self._environment._goal:
            self._goal_collected = True
        self._distance_to_goal = self.calculate_distance_to_goal()

        # Shift weight, increase weight shifted, remove weight from old position
        self._environment._environment[new_shift_pos[0]][new_shift_pos[1]] += self._environment._environment[new_pos[0]][new_pos[1]]
        self._weight_shifted += self._environment._environment[new_pos[0]][new_pos[1]]
        self._environment._environment[new_pos[0]][new_pos[1]] = 0

    def get_all_valid_pairs(self) -> list[tuple]:
        """Returns a list of all valid movement and shfiting directions pairs at the current position.

        Returns:
            list[tuple]: List of valid pairs.
        """
        valid_pairs = []

        cx, cy = self._current_pos
        dim = self._environment._env_dim

        dirs = tuple(Direction)
        dir_vals = [d.value for d in dirs]

        # Outer loop: movement direction
        for i, (dx_m, dy_m) in enumerate(dir_vals):
            x_m = cx + dx_m
            y_m = cy + dy_m

            # bounds check #1
            if x_m < 0 or x_m >= dim or y_m < 0 or y_m >= dim:
                continue

            move_dir = dirs[i]  # retrieve Enum only once

            # Inner loop: shifting direction
            for j, (dx_s, dy_s) in enumerate(dir_vals):
                x_s = x_m + dx_s
                y_s = y_m + dy_s

                # bounds check #2
                if x_s < 0 or x_s >= dim or y_s < 0 or y_s >= dim:
                    continue

                shift_dir = dirs[j]
                valid_pairs.append((move_dir, shift_dir))

        return valid_pairs

    def is_valid_pair(self, pair: tuple[Direction, Direction]) -> bool:
        """Check if a movement and shifting direction pair is valid."""
        move_dir, shift_dir = pair
        dim = self._environment._env_dim
        x, y = self._current_pos

        # Apply move
        dx_m, dy_m = move_dir.value
        x_m, y_m = x + dx_m, y + dy_m

        # Apply shift after move
        dx_s, dy_s = shift_dir.value
        x_s, y_s = x_m + dx_s, y_m + dy_s

        # Inline boundary checks (avoid function calls)
        return (
            0 <= x_m < dim and 0 <= y_m < dim and
            0 <= x_s < dim and 0 <= y_s < dim
        )

    def is_valid_positon(self, position: tuple) -> bool:
        """Checks if position is inside the map bounds.

        Args:
            position (tuple): Position to be checked

        Returns:
            bool: Is position valid or not
        """

        dim = self._environment._env_dim
        x, y = position
        
        return 0 <= x < dim and 0 <= y < dim

    def calculate_distance_to_goal(self) -> float:
        """Calculates the Manhattan distance to the goal from the current position.

        Returns:
            float: Manhattan distance to goal
        """
        current_pos = self._current_pos
        goal = self._environment._goal
        start = self._start_pos
        goal_collected = self._goal_collected

        # Manual Manhattan dist (faster than lambda + tuple unpack)
        dx1 = current_pos[0] - goal[0]
        dy1 = current_pos[1] - goal[1]
        dist_to_goal = (dx1 if dx1 >= 0 else -dx1) + (dy1 if dy1 >= 0 else -dy1)

        if not goal_collected:
            dx2 = start[0] - goal[0]
            dy2 = start[1] - goal[1]
            return dist_to_goal + (dx2 if dx2 >= 0 else -dx2) + (dy2 if dy2 >= 0 else -dy2)
        else:
            dx3 = current_pos[0] - start[0]
            dy3 = current_pos[1] - start[1]
            return (dx3 if dx3 >= 0 else -dx3) + (dy3 if dy3 >= 0 else -dy3)
