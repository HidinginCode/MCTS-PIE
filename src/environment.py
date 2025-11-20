"""This module contains the environment class, which holds the obstacle map and dimensions."""

from __future__ import annotations
import random
import os
import pickle

class Environment():
    """This class represents the obstacle environment."""

    def __init__(self, env_dim: int = 10, goal: tuple = (9,9), new_env: bool = True, map_type: str = None) -> None:
        """Init method for the environment which sets the map dimension and the goal.

        Args:
            env_dim (int, optional): Environment dimension. Defaults to 10.
            goal (tuple, optional): Goal position. Defaults to (9,9).
        """
        
        self._env_dim = env_dim
        self._goal = goal
        self._identifier = id(self)

        if not os.path.exists("./maps"):
            self.generate_maps(env_dim)
        
        if map_type is not None or new_env:
            with open(f"./maps/{map_type}_{env_dim}x{env_dim}.pickle", "rb") as f:
                self._environment = pickle.load(f)
        else:
            self._environment = []
        
        self._map_type = map_type

    def clone(self) -> Environment:
        """Creates an independent clone of the environment object.

        Returns:
            Environment: Cloned environment
        """
        cloned_env = Environment(env_dim=self.env_dim, goal=self.goal, new_env=False)
        cloned_env._environment = [row[:] for row in self._environment]  # FAST & correct
        cloned_env._map_type = self._map_type
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
    
    def generate_maps(self, env_dim: int = 10):
        """This method generates AND safes maps to a directory.

        Args:
            env_dim (int, optional): Defines the environment size. Defaults to 10.
        """
        print("Generating maps according to speicifcations ...")
        map_path = "./maps"
        if not os.path.exists(map_path):
            os.mkdir(map_path)

        #######################
        # Generate Random Map #
        #######################
        random_map = [[random.random() if (x,y) != (0,0) else 0 for y in range(env_dim)] for x in range(env_dim)]

        #############################
        # Generate Checkerboard Map #
        #############################
        checkerboard_map = [[(random.random() if (x, y) != (0, 0) and (x + y) % 2 == 1 else 0.0) for y in range(env_dim)]for x in range(env_dim)]

        ##################################
        # Generate Map with Obvious Path #
        ##################################
        sx, sy = (0,0)
        gx, gy = (env_dim-1, env_dim-1)

        easy_map = [[random.random() if (x,y) != (0,0) else 0 for y in range(env_dim)] for x in range(env_dim)]

        x, y = sx, sy
        easy_map[x][y] = 0

        def manhattan(a, b):
            return abs(a[0]-b[0]) + abs(a[1]-b[1])

        while (x, y) != (gx, gy):
            candidates = []
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:  # von Neumann
                nx, ny = x + dx, y + dy
                if 0 <= nx < env_dim and 0 <= ny < env_dim:
                    if manhattan((nx, ny), (gx, gy)) < manhattan((x, y), (gx, gy)):
                        candidates.append((nx, ny))

            # pick randomly among distance-reducing moves
            x, y = random.choice(candidates)
            easy_map[x][y] = 0
        
        # Save all maps to files
        with open(os.path.join(map_path, f"random_map_{env_dim}x{env_dim}.pickle"), "wb") as f:
            pickle.dump(random_map, f)

        with open(os.path.join(map_path, f"checkerboard_map_{env_dim}x{env_dim}.pickle"), "wb") as f:
            pickle.dump(checkerboard_map, f)

        with open(os.path.join(map_path, f"easy_map_{env_dim}x{env_dim}.pickle"), "wb") as f:
            pickle.dump(easy_map, f)