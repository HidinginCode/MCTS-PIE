"""This module contains the environment class, which holds the obstacle map and dimensions."""

from __future__ import annotations
import numpy as np

class Environment():
    """This class represents the obstacle environment."""

    def __init__(self, env_dim: int = 10, goal: tuple = (9,9)) -> None:
        """Init method for the environment which sets the map dimension and the goal.

        Args:
            env_dim (int, optional): Environment dimension. Defaults to 10.
            goal (tuple, optional): Goal position. Defaults to (9,9).
        """
        
        self._env_dim = env_dim
        self._goal = goal
        self._identifier = id(self)
        self._environment = np.random.random((env_dim, env_dim))

    def clone(self) -> Environment:
        """Creates an independent clone of the environment object.

        Returns:
            Environment: Cloned environment
        """
        cloned_env = Environment(env_dim=self.env_dim, goal=self.goal)
        cloned_env._environment = self._environment.copy()
        return cloned_env

    @property
    def env_dim(self) -> int:
        """Getter for env_dim argument.

        Returns:
            int: Environment dimension
        """
        return self._env_dim

    @property
    def environment(self) -> np.ndarray:
        """Getter for environment.

        Returns:
            np.ndarray: Environment array
        """
        return self._environment
    
    @environment.setter
    def environment(self, environment: np.ndarray) -> None:
        """Sets the environment array.

        Args:
            environment (np.ndarray): Environment array
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