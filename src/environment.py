"""This module contains the environment class, which holds the obstacle map and dimensions."""

from __future__ import annotations
import random

class Environment():
    """This class represents the obstacle environment."""

    def __init__(self, env_dim: int = 10, goal: tuple = (9,9), new_env: bool = True) -> None:
        """Init method for the environment which sets the map dimension and the goal.

        Args:
            env_dim (int, optional): Environment dimension. Defaults to 10.
            goal (tuple, optional): Goal position. Defaults to (9,9).
        """
        
        self._env_dim = env_dim
        self._goal = goal
        self._identifier = id(self)
        if new_env:
            self._environment = [[random.random() if (x,y) != (0,0) else 0 for y in range(env_dim)] for x in range(env_dim)]
        else:
            self._environment = []

    def clone(self) -> Environment:
        """Creates an independent clone of the environment object.

        Returns:
            Environment: Cloned environment
        """
        cloned_env = Environment(env_dim=self.env_dim, goal=self.goal, new_env=False)
        cloned_env._environment = [row[:] for row in self._environment]  # FAST & correct
        return cloned_env

    @property
    def env_dim(self) -> int:
        """Getter for env_dim argument.

        Returns:
            int: Environment dimension
        """
        return self._env_dim

    @property
    def environment(self) -> list:
        """Getter for environment.

        Returns:
            list: Environment array
        """
        return self._environment
    
    @environment.setter
    def environment(self, environment: list) -> None:
        """Sets the environment array.

        Args:
            list: Environment array
        """
        self._environment = environment.copy()

    @property
    def identifier(self) -> int:
        """Getter for _identifier.

        Returns:
            int: Environment ID
        """
        return self._identifier

    @property
    def goal(self) -> tuple:
        """Getter for goal of environment.

        Returns:
            tuple: Goal of environment.
        """
        return self._goal