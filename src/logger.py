"""This module contains the logger class, responsible for collecting data."""

import os
import shutil
from node import Node
from analyzer import Analyzer
import pandas as pd
import pickle
from time import time

class Logger():
    """Logger class which logs information for each run and prepares directories"""

    def __init__(self, 
                 map_name: str,
                 env_dim: int,
                 start: tuple,
                 goal: tuple,
                 total_budget: int,
                 per_sim_budget: int,
                 number_of_simulations: int, 
                 tree_selection_method: str,
                 root_selection_method: str, 
                 max_pareto_paths: int,
                 simulation_method: str,
                 seed: int,
                 base_root: Node,
                 base_log_path: str = "./log"):
        self._base_log_path = base_log_path
        self._map_name = map_name
        self._total_budget = total_budget
        self._per_sim_budget = per_sim_budget
        self._number_of_simulations = number_of_simulations
        self._tree_selection_method = tree_selection_method
        self._root_selection_method = root_selection_method
        self._max_pareto_paths = max_pareto_paths
        self._simulation_method = simulation_method
        self._env_dim = env_dim
        self._start = start
        self._goal = goal
        self._seed = seed
        self._root = base_root

        if not os.path.exists(base_log_path):
            os.mkdir(base_log_path)
        
        dir_name = f"{self._map_name}-{self._env_dim}-{self._start}-{self._goal}-{self._tree_selection_method}-{self._root_selection_method}-{self._simulation_method}-{self._total_budget}-{self._per_sim_budget}-{self._number_of_simulations}-{self._max_pareto_paths}-{self._seed}"
        log_path = os.path.join(base_log_path, dir_name)
        if os.path.exists(log_path):
            shutil.rmtree(log_path)

        os.mkdir(log_path)

        self._log_path = log_path

    def log_solutions(self, solutions: list[Node]) -> None:
        """Log the solutions of a run in log directory.

        Args:
            solutions (list): Solutions of the run.
        """
        for i, solution in enumerate(solutions):
            solution: Node
            current = solution
            path = []
            shifts = []
            moves = []

            # Path reconstruction

            while current is not None:
                path.append(current._controller._current_pos)
                if current._last_move is not None:#
                    moves.append(current._last_move)
                    shifts.append(current._last_move[1])
                current = current._parent
            
            moves.reverse()
            shifts.reverse()
            path.reverse()

            Analyzer.visualize_path_with_shifts(solution._controller._environment._environment, path, shifts, (0,0), solution._controller._environment._goal, f"{self._log_path}/path-{i}.png")
            Analyzer.save_path_as_gif(self._root._controller._environment, self._root._controller._start_pos, moves, f"{self._log_path}/path-{i}.gif")

            solution.refresh_values()
            data = solution._values.copy()
            data["path"] = path
            data["moves"] = moves

            with open(f"{self._log_path}/{i}-values.pickle", "wb") as f:
                pickle.dump(data,f)

