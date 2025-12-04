"""This module contains the logger class, responsible for collecting data."""

import os
import pickle
from node import Node

class Logger:
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

        self._map_name = map_name
        self._env_dim = env_dim
        self._start = start
        self._goal = goal
        self._total_budget = total_budget
        self._per_sim_budget = per_sim_budget
        self._number_of_simulations = number_of_simulations
        self._tree_selection_method = tree_selection_method
        self._root_selection_method = root_selection_method
        self._max_pareto_paths = max_pareto_paths
        self._simulation_method = simulation_method
        self._seed = seed
        self._root = base_root

        # -----------------------------
        #  SAFE DIRECTORY CREATION
        # -----------------------------
        os.makedirs(base_log_path, exist_ok=True)

        # Directory WITHOUT seed
        dir_name = (
            f"{map_name}-{env_dim}-{start}-{goal}-"
            f"{tree_selection_method}-{root_selection_method}-"
            f"{simulation_method}-{total_budget}-{per_sim_budget}-"
            f"{number_of_simulations}-{max_pareto_paths}"
        )

        self._log_path = os.path.join(base_log_path, dir_name)

        # race-safe directory creation for multiple processes
        os.makedirs(self._log_path, exist_ok=True)

        # Header is written ONCE — guard with exist check
        header_path = os.path.join(self._log_path, "header.pickle")
        if not os.path.exists(header_path):
            self.create_header_file(header_path)

    # ----------------------------------------------------------
    #   LOG SOLUTIONS → writes ONLY <seed>.pickle
    #   SAFE FOR MULTIPROCESS EXECUTION
    # ----------------------------------------------------------
    def log_solutions(self, solutions: list[Node]) -> None:

        collected = []

        for solution in solutions:
            current = solution
            path = []
            moves = []
            shifts = []

            # Reconstruct path
            while current is not None:
                path.append(current._controller._current_pos)

                if current._last_move is not None:
                    move, shift = current._last_move
                    moves.append(move)
                    shifts.append(shift)

                current = current._parent

            # Reverse once
            path.reverse()
            moves.reverse()
            shifts.reverse()

            solution.refresh_values()

            data = {
                "values": solution._values.copy(),
                "path": path,
                "moves": moves,
                "shifts": shifts,
            }

            collected.append(data)

        # Write only to <seed>.pickle → no race conditions
        out_path = os.path.join(self._log_path, f"{self._seed}.pickle")
        with open(out_path, "wb") as f:
            pickle.dump(collected, f)

    # ----------------------------------------------------------
    # HEADER FILE WRITTEN ONCE
    # ----------------------------------------------------------
    def create_header_file(self, header_path: str) -> None:
        header_dict = {
            "map_name": self._map_name,
            "env_dim": self._env_dim,
            "start": self._start,
            "goal": self._goal,
            "total_budget": self._total_budget,
            "per_sim_budget": self._per_sim_budget,
            "number_of_simulations": self._number_of_simulations,
            "tree_selection_method": self._tree_selection_method,
            "root_selection_method": self._root_selection_method,
            "max_pareto_paths": self._max_pareto_paths,
            "simulation_method": self._simulation_method,
        }

        # Safe write of header
        with open(header_path, "wb") as f:
            pickle.dump(header_dict, f)
