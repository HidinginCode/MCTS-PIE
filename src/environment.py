"""This module contains the environment class, which holds the obstacle map and dimensions."""

from __future__ import annotations
import random
import os
import pickle
import numpy as np
from scipy.ndimage import gaussian_filter

class Environment():
    """This class represents the obstacle environment."""

    def __init__(self, env_dim: int = 10, goal: tuple = (9,9), new_env: bool = True, map_type: str = None, start_pos: tuple = (0,0), checkpoints: list = None) -> None:
        """Init method for the environment which sets the map dimension and the goal.

        Args:
            env_dim (int, optional): Environment dimension. Defaults to 10.
            goal (tuple, optional): Goal position. Defaults to (9,9).
            checkpoints (list, optional): Ordered list of checkpoint positions the
                agent must traverse before returning to ``start_pos``. When
                ``None`` (default), the environment behaves as before with
                ``goal`` as the sole checkpoint.
        """

        self._env_dim = env_dim
        self._goal = goal
        self._identifier = id(self)
        self._start_pos = start_pos

        # Canonicalize checkpoints: single-goal default mirrors pre-refactor
        # behavior. Explicitly provided lists are copied so mutating the
        # argument later does not leak into the Environment state.
        if checkpoints is None:
            self._checkpoints = [tuple(goal)]
        else:
            self._checkpoints = [tuple(cp) for cp in checkpoints]

        if not os.path.exists("./maps"):
            self.generate_maps(env_dim)

        if map_type is not None or new_env:
            with open(f"./maps/{map_type}_{env_dim}x{env_dim}.pickle", "rb") as f:
                self._environment = pickle.load(f)
        else:
            self._environment = []

        self._map_type = map_type

    def sample_checkpoints(self, n: int, seed: int) -> list[tuple[int, int]]:
        """Deterministically draw ``n`` distinct corridor cells as checkpoints.

        Cells with weight strictly less than 0.5 are considered corridor; the
        agent's start, goal, and any duplicates are excluded. Returns an
        ordered list usable as the ``checkpoints`` argument.
        """
        rng = random.Random(seed)
        forbidden = {tuple(self._start_pos), tuple(self._goal)}
        candidates = [
            (x, y)
            for x in range(self._env_dim)
            for y in range(self._env_dim)
            if (x, y) not in forbidden and self._environment[x][y] < 0.5
        ]
        rng.shuffle(candidates)
        return [tuple(c) for c in candidates[:n]]

    def clone(self) -> Environment:
        """Creates an independent clone of the environment object.

        Returns:
            Environment: Cloned environment
        """
        cloned_env = Environment(env_dim=self.env_dim, goal=self.goal, new_env=False)
        cloned_env._environment = [row[:] for row in self._environment]  # FAST & correct
        cloned_env._map_type = self._map_type
        cloned_env._start_pos = self._start_pos
        cloned_env._checkpoints = list(self._checkpoints)
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
    
    def generate_maps(self, env_dim_old: int = 10):
        """This method generates AND safes maps to a directory.

        Args:
            env_dim (int, optional): Defines the environment size. Defaults to 10.
        """
        env_dims = (20, 35, 50)
        for env_dim in env_dims:
            print("Generating maps according to speicifcations ...")
            map_path = "./maps"
            if not os.path.exists(map_path):
                os.mkdir(map_path)

            #######################
            # Generate Random Map #
            #######################
            random_map = [[random.random() if (x,y) != self._start_pos else 0 for y in range(env_dim)] for x in range(env_dim)]
            random_map[0][env_dim//2] = 0
            #############################
            # Generate Checkerboard Map #
            #############################
            x = np.linspace(0, 5 * np.pi, env_dim)
            y = np.linspace(0, 5 * np.pi, env_dim)
            x, y = np.meshgrid(x, y)
            # combining sine and cosine functions
            checkerboard_map= np.sin(x) * np.cos(y)
            # Normalize to 0-1 range
            checkerboard_map = (checkerboard_map - checkerboard_map.min()) / (checkerboard_map.max() - checkerboard_map.min())
            checkerboard_map = checkerboard_map.tolist()
            checkerboard_map[0][env_dim//2] = 0

            ##################################
            # Generate Map with Obvious Path #
            ##################################
            sx, sy = (0, env_dim//2)
            gx, gy = (env_dim-1, env_dim//2)

            easy_map = [[random.random() if (x,y) != self._start_pos else 0 for y in range(env_dim)] for x in range(env_dim)]

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

            meandering_river = np.zeros((env_dim, env_dim))

            # Parameters for the river path
            river_width = 7
            t = np.linspace(0, 1, env_dim)
            x_center = env_dim / 2
            amplitude = env_dim / 3

            # Generate smooth S-shaped path
            x_path = x_center + amplitude * np.sin(2 * np.pi * t)

            # Mark the river path on the obstacle map
            for y, x in enumerate(x_path):
                x_start = int(x - river_width / 2)
                x_end = int(x + river_width / 2)
                meandering_river[y, max(x_start, 0):min(x_end, env_dim)] = 1

            # Apply Gaussian filter to create gradient effect
            meandering_river = gaussian_filter(meandering_river, sigma=3)

            # Invert colors: black parts white and white parts black
            meandering_river = 1 - meandering_river

            # Normalize to 0-1 range
            meandering_river = (meandering_river - np.min(meandering_river)) / (np.max(meandering_river) - np.min(meandering_river))
            meandering_river = meandering_river.tolist()
            meandering_river[0][env_dim//2] = 0


            # ############################
            # # Random Maze Map
            # ############################
            # A randomized-DFS maze carved on odd cells, then extra openings
            # are punched through walls so there are multiple paths between
            # most corridor pairs (making weight-shifting non-trivial).
            random_maze = self._generate_random_maze(env_dim)

            # Save all maps to files
            with open(os.path.join(map_path, f"random_map_{env_dim}x{env_dim}.pickle"), "wb") as f:
                pickle.dump(random_map, f)

            with open(os.path.join(map_path, f"checkerboard_map_{env_dim}x{env_dim}.pickle"), "wb") as f:
                pickle.dump(checkerboard_map, f)

            with open(os.path.join(map_path, f"easy_map_{env_dim}x{env_dim}.pickle"), "wb") as f:
                pickle.dump(easy_map, f)

            with open(os.path.join(map_path, f"meandering_river_map_{env_dim}x{env_dim}.pickle"), "wb") as f:
                pickle.dump(meandering_river, f)

            with open(os.path.join(map_path, f"random_maze_{env_dim}x{env_dim}.pickle"), "wb") as f:
                pickle.dump(random_maze, f)

    @staticmethod
    def _generate_random_maze(env_dim: int, extra_opening_frac: float = 0.2) -> list:
        """Generate a random maze with multiple paths between corridor cells.

        Walls are stored as high-weight cells (1.0 plus small noise), corridors
        as low-weight cells (near 0.0). After carving with randomized DFS, a
        fraction of walls is knocked out to create loops — this gives the
        weight-shifting agent a genuine choice of routes.

        Returns:
            list[list[float]]: Square grid of shape (env_dim, env_dim).
        """
        walls = [[1 for _ in range(env_dim)] for _ in range(env_dim)]

        # Randomized DFS on odd (x, y) cells.
        start = (1, 1) if env_dim > 2 else (0, 0)
        walls[start[0]][start[1]] = 0
        stack = [start]
        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < env_dim and 0 <= ny < env_dim and walls[nx][ny] == 1:
                    neighbors.append((nx, ny, dx, dy))
            if not neighbors:
                stack.pop()
                continue
            nx, ny, dx, dy = random.choice(neighbors)
            walls[cx + dx // 2][cy + dy // 2] = 0
            walls[nx][ny] = 0
            stack.append((nx, ny))

        # Punch out a fraction of inner walls to create multiple paths.
        candidate_walls = [
            (x, y)
            for x in range(1, env_dim - 1)
            for y in range(1, env_dim - 1)
            if walls[x][y] == 1
        ]
        random.shuffle(candidate_walls)
        to_open = int(len(candidate_walls) * extra_opening_frac)
        for x, y in candidate_walls[:to_open]:
            walls[x][y] = 0

        # Convert to continuous weights: corridors ~ 0.0, walls ~ 1.0 with noise.
        maze = [
            [
                (0.0 + 0.05 * random.random()) if walls[x][y] == 0 else (1.0 - 0.1 * random.random())
                for y in range(env_dim)
            ]
            for x in range(env_dim)
        ]
        # Ensure start and end corners of the middle column are open.
        maze[0][env_dim // 2] = 0
        maze[env_dim - 1][env_dim // 2] = 0
        return maze