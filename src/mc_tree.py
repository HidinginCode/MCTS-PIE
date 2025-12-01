"""This module contains the MCTS tree."""

from __future__ import annotations
from node import Node
from helper import Helper
from controller import Controller
from environment import Environment
import multiprocessing as mp
import os
import random
import numpy as np
import copy
from logger import Logger
import math


class MctsTree():
    """This class contains the mcts tree."""

    def __init__(self, root: Node, seed: int,  max_solutions: int = 10):
        """Init method for the MCTS tree.

        Args:
            root (Node): Root node for the tree
            seed (int): Seed (in this case only important to pass to logger)
            max_solutions (int): Maximum number of entries in pareto front
        """
        self._identifier = id(self)
        self._root = root
        self._max_solutions = max_solutions
        self._max_depth = 0
        self._seed = seed

    @property
    def identifier(self) -> int:
        """Getter for tree identifier.

        Returns:
            int: ID of tree
        """
        return self._identifier

    @property
    def root(self) -> Node:
        """Getter for root of MCTS tree.

        Returns:
            Node: Root node of tree
        """
        return self._root

    def tree_policy(self, root: Node, tree_sel_func: function) -> Node | None:
        """Tree policy that selects the path from the root to a leaf.

        Args:
            root: Root to start policy from
            tree_sel_func: Function used to select next node in tree

        Returns:
            Node | None: Either leaf node or none if i.e. solution was selected.
        """

        current_node = root
        while True:
            
            # Check if the node has already reached the goal
            if current_node.is_terminal_state():
                return None

            # Check if the node is fully expanded
            if current_node.get_untried_actions():
                return current_node

            # At this point we know that current_node is neither terminal nor has any expansion left
            # Safety check for children
            if current_node._children:
                current_node = tree_sel_func(current_node)

    def pareto_path_child_selection_cd(self, node: Node) -> Node:
        """Method that selects a child from pareto_paths using the crowding distance.

        Args:
            node (Node): Node of which to select a child
        """

        if not node._pareto_paths:
            return random.choice(list(node._children.values()))

        # Recompute CDs only if front changed
        if node._paths_changed:
            node._paths_changed = False
            values = [path[0][1] for path in node._pareto_paths]
            crowding_distances = Helper.crowding_distance(values)
            node._old_cd_values = crowding_distances
        else:
            crowding_distances = node._old_cd_values

        cds = crowding_distances
        n = len(cds)

        # Identify extremes (inf) and finite CDs
        inf_indices = [i for i, c in enumerate(cds) if c == np.inf]
        finite_indices = [i for i, c in enumerate(cds) if np.isfinite(c)]
        finite_values = [cds[i] for i in finite_indices]

        weights = [0.0] * n

        # Case 1: no extremes -> normal crowding-distance selection
        if not inf_indices:
            total = sum(finite_values)
            if total <= 0:
                # all zeros -> uniform
                weights = [1.0] * n
            else:
                for i in finite_indices:
                    weights[i] = cds[i] / total

        else:
            # We *do* have extremes: give them 50% of the mass, the rest to finite ones
            extreme_mass = 0.5
            finite_mass = 0.5

            # If there are no finite positive CDs, just pick among extremes
            finite_sum = sum(finite_values)
            if finite_sum <= 0:
                # Only extremes matter -> uniform over extremes
                for i in inf_indices:
                    weights[i] = 1.0
            else:
                # Distribute 50% equally among extremes
                per_extreme = extreme_mass / len(inf_indices)
                for i in inf_indices:
                    weights[i] = per_extreme

                # Distribute the other 50% proportional to finite CDs
                for i in finite_indices:
                    weights[i] += (cds[i] / finite_sum) * finite_mass

        # Now sample according to weights
        child_key_index = random.choices(range(n), weights=weights, k=1)[0]
        child_key = node._pareto_paths[child_key_index][-1][0]

        return node._children[child_key]

    def pareto_path_child_selection_hv(self, node: Node) -> Node:
        """Method that selects children based on stored pareto paths based on hypervolume.

        Args:
            node (Node): Root node for child selection

        Returns:
            Node: Child node
        """

        # Check if there are pareto paths
        if not node._pareto_paths:
            return random.choice(node._children.values())
        
        if node._paths_changed:
            values = [path[0][1] for path in node._pareto_paths]
            hypervolumes, full_hv = Helper.hypervolume(values)
            # Since paths changed we save new hypervolumes
            node._old_hv_values = (hypervolumes, full_hv)
            #print(f"Node old hv values: {node._old_hv_values}")
            node._paths_changed = False
        else:
            hypervolumes, full_hv = node._old_hv_values

        weights = [hv/full_hv for hv in hypervolumes]
        child_key_index = random.choices(range(len(hypervolumes)), weights, k=1)[0]
        child_key = node._pareto_paths[child_key_index][-1][0]

        return node._children[child_key]

    def ucb_child_selection(self, node: Node) -> Node:
        """Select children based on Pareto dominance of normalized UCB1 values (minimization)."""
        children = list(node._children.values())
        if not children:
            raise RuntimeError("ucb_child_selection called on node without children")

        dims = list(children[0]._values.keys())

        mins = {d: float("inf") for d in dims}
        maxs = {d: float("-inf") for d in dims}
        for child in children:
            for d, v in child._values.items():
                if v < mins[d]:
                    mins[d] = v
                if v > maxs[d]:
                    maxs[d] = v

        # Avoid zero-range problems
        ranges = {d: (maxs[d] - mins[d]) if (maxs[d] > mins[d]) else 1.0 for d in dims}

        parent_visits = max(1, node._visits)
        logN = np.log(parent_visits)
        c = 1.0  # exploration coefficient

        for child in children:
            if child._visits == 0:
                # force min priority on all objectives → guaranteed exploration
                child._ucb_values = {d: -np.inf for d in dims}
                continue

            # normalized values
            norm = {
                d: (child._values[d] - mins[d]) / ranges[d]
                for d in dims
            }

            # multi-objective adaptive exploration penalties
            # distance gets weaker exploration (since it already drives)
            alpha = {
                "distance_to_goal": 0.3,
                "step_count": 1.0,
                "weight_shifted": 1.0,
            }

            for d in dims:
                explore = alpha[d] * np.sqrt(2 * logN / child._visits)
                child._ucb_values[d] = norm[d] - explore

        # Determine Pareto front in UCB space and pick one child
        pareto_front = Helper.determine_pareto_front_from_nodes(children, True)
        return random.choice(pareto_front)

    def light_rollout(self, leaf: Node, simulations: int, maximum_moves: int, remaining_budget: int)  -> int:
        """Light rollout which does random moves.

        Args:
            leaf (Node): Leaf from which to simulate.
            simulations (int): Number of iterative simulations.
            maximum_moves (int): Maximum move per simulation.
            remaining_budget (int): Remaining step budget for simulations.

        Returns:
            int: Used step budget
        """

        results = []
        used_move_counter = 0

        for _ in range(simulations):
            # Clone so we have independent simulation
            leaf_copy = leaf.clone()
            controller = leaf_copy._controller

            for _ in range(maximum_moves):
                # Break if we reached terminal state or used whole budget
                if leaf_copy.is_terminal_state() or used_move_counter >= remaining_budget:
                    break
                # Choose randomly from all valid pairs
                all_valid_moves = controller.get_all_valid_pairs()
                move_dir, shift_dir = random.choice(all_valid_moves)
                controller.move(move_dir, shift_dir)

                # Update move counter
                used_move_counter += 1
            
            results.append(leaf_copy.clone())

            if used_move_counter >= remaining_budget:
                break
        
        chosen_node = random.choice(Helper.determine_pareto_front_from_nodes(results))
        leaf._values = chosen_node._values.copy()
        return used_move_counter


    def iterative_heavy_angular_rollout(self, leaf: Node, simulations: int, maximum_moves: int, remaining_budget: int) -> int:
        """
        Iterative rollout method that:
        1) Samples points in a square radius around the current position
        2) Chooses a sampled point that is closer to the goal than the current position
        3) If none exists → fallback to von Neumann neighbors (just like heavy_distance_rollout)

        Args:
            leaf (Node): Leaf node to simulate from.
            simulations (int): Number of rollouts.
            maximum_moves (int): Max moves per rollout.
            remaining_budget (int): Remaining global budget.

        Returns:
            int: total moves used across rollout
        """

        radius = max(2, min(8, leaf._controller._environment.env_dim // 6))

        # Precomputed directional vectors (8  directions)
        DIRS = [
            (1, 0), (1, 1), (0, 1), (-1, 1),
            (-1, 0), (-1, -1), (0, -1), (1, -1)
        ]

        # How many directions to sample
        NUM_DIR_SAMPLES = 4

        results = []
        used_move_counter = 0

        for _ in range(simulations):

            leaf_copy = leaf.clone()
            controller = leaf_copy._controller
            env_dim = controller._environment.env_dim
            start = controller._start_pos
            goal = controller._environment._goal

            def manhattan(a, b):
                dx = a[0] - b[0]
                dy = a[1] - b[1]
                return (dx if dx >= 0 else -dx) + (dy if dy >= 0 else -dy)

            for _ in range(maximum_moves):

                # Stop if terminal or out of budget
                if leaf_copy.is_terminal_state() or used_move_counter >= remaining_budget:
                    break

                cx, cy = controller._current_pos
                gx, gy = (start if controller._goal_collected else goal)
                current_dist = manhattan((cx, cy), (gx, gy))

                
                # sample only directions that point toward the goal
                vgx = gx - cx
                vgy = gy - cy

                filtered_dirs = []
                for dx, dy in DIRS:
                    # direction should point toward goal (dot product > 0)
                    dot = dx * vgx + dy * vgy
                    if dot <= 0:
                        continue

                    # remove purely horizontal/vertical collinearity
                    #if (dx == 0 and vgx == 0) or (dy == 0 and vgy == 0):
                    #    continue

                    filtered_dirs.append((dx, dy))

                # if filtering removed everything, fallback to all directions
                if not filtered_dirs:
                    filtered_dirs = DIRS

                # choose a few directions to sample
                chosen_dirs = random.sample(filtered_dirs, min(NUM_DIR_SAMPLES, len(filtered_dirs)))

                # sample targets using directions
                better_samples = []
                for dx, dy in chosen_dirs:
                    tx = cx + dx * radius
                    ty = cy + dy * radius

                    if 0 <= tx < env_dim and 0 <= ty < env_dim:
                        new_dist = manhattan((tx, ty), (gx, gy))
                        if new_dist < current_dist:
                            better_samples.append((tx, ty, new_dist))

                # If we found good sample, greedily move toward best one
                if better_samples:
                    better_samples.sort(key=lambda t: t[2])
                    target_x, target_y, _ = better_samples[0]

                    valid_pairs = controller.get_all_valid_pairs()
                    best_pair = None
                    best_dist = float("inf")

                    for move_dir, shift_dir in valid_pairs:
                        nx = cx + move_dir[0]
                        ny = cy + move_dir[1]
                        d = manhattan((nx, ny), (target_x, target_y))
                        if d < best_dist:
                            best_dist = d
                            best_pair = (move_dir, shift_dir)

                    controller.move(best_pair[0], best_pair[1])
                    used_move_counter += 1
                    continue


                # Fallback -> greedy toward goal
                valid_pairs = controller.get_all_valid_pairs()
                best_pair = None
                best_goal_dist = float("inf")

                for move_dir, shift_dir in valid_pairs:
                    nx = cx + move_dir[0]
                    ny = cy + move_dir[1]
                    d = manhattan((nx, ny), (gx, gy))
                    if d < best_goal_dist:
                        best_goal_dist = d
                        best_pair = (move_dir, shift_dir)

                if best_pair is not None:
                    controller.move(best_pair[0], best_pair[1])
                    used_move_counter += 1
                    continue

                # Last fallback ->  random move
                move_dir, shift_dir = random.choice(valid_pairs)
                controller.move(move_dir, shift_dir)
                used_move_counter += 1

            results.append(leaf_copy.clone())
            if used_move_counter >= remaining_budget:
                break

        chosen_node = random.choice(Helper.determine_pareto_front_from_nodes(results))
        leaf._values = dict(chosen_node._values)
        return used_move_counter

    def iterative_heavy_distance_weight_rollout(self, leaf: Node, simulations: int, maximum_moves: int, remaining_budget: int) -> int:
        """Iterative version of the heavy distance weight rollout to look into performance.

        Args:
            leaf (Node): Leaf to simulate.
            simulations (int): Number of iterative simulations.
            maximum_moves (int): Maximum number of moves per simulation.
            remaining_budget (int): Remaining simulation budget.
        
        Returns:
            Number of moves used for simulation
        """
        results = []
        used_move_counter = 0
        for _ in range(simulations):
            # Clone so we have independent simulation
            leaf_copy = leaf.clone()
            controller = leaf_copy._controller
            start = controller._start_pos
            goal = controller._environment._goal
            env = controller._environment._environment

            # Precompute constant
            dxg = start[0] - goal[0]
            dyg = start[1] - goal[1]
            roundtrip_back = (dxg if dxg >= 0 else -dxg) + (dyg if dyg >= 0 else -dyg)

            def distance(pos):
                x, y = pos

                if controller._goal_collected:
                    gx, gy = start
                else:
                    gx, gy = goal

                dx = x - gx
                dy = y - gy

                # Branchless abs via comparison
                return (dx if dx >= 0 else -dx) + (dy if dy >= 0 else -dy)


            for _ in range(maximum_moves):

                # Break if we reached terminal state
                if leaf_copy.is_terminal_state() or used_move_counter >= remaining_budget:
                    break

                # Get needed parts of calculation and prepare move list
                current_pos = controller._current_pos
                current_distance_to_goal = distance(current_pos)
                distance_minimizing_moves = []
                weight_for_distance_min_moves = []
                valid_moves = leaf_copy._controller.get_all_valid_pairs()
                if not controller._goal_collected:
                    alligned_axis = current_pos[0] == goal[0] or current_pos[1] == goal[1]
                else:
                    alligned_axis = current_pos[0] == start[0] or current_pos[1] == start[1]

                # Get moves that do not increase distance
                slack = 1 # Allows for moves that are non optimal -> breaks problem that we have with straigt paths on alligned objectives
                for move_dir, shifting_dir in valid_moves:
                    new_pos = (current_pos[0] + move_dir[0],
                            current_pos[1] + move_dir[1])
                    new_distance_to_goal = distance(new_pos)
                    if alligned_axis: # Allow slack when axis alligned to prevent straight paths
                        if new_distance_to_goal <= current_distance_to_goal + slack:
                            distance_minimizing_moves.append((move_dir, shifting_dir))
                            weight_for_distance_min_moves.append(env[new_pos[0]][new_pos[1]])
                    else:
                        if new_distance_to_goal <= current_distance_to_goal:
                            distance_minimizing_moves.append((move_dir, shifting_dir))
                            weight_for_distance_min_moves.append(env[new_pos[0]][new_pos[1]])

                # Choose move that has least weight
                min_index = min(range(len(weight_for_distance_min_moves)), key=weight_for_distance_min_moves.__getitem__)
                move_direction, shift_direction = distance_minimizing_moves[min_index]
                controller.move(move_direction, shift_direction)
                used_move_counter += 1

            results.append(leaf_copy.clone())

            if used_move_counter >= remaining_budget:
                break
        
        chosen_node = random.choice(Helper.determine_pareto_front_from_nodes(results))
        leaf._values = dict(chosen_node._values)
        return used_move_counter

    @staticmethod
    def path_domination(path1: list, path2: list) -> bool:
        """Returns if path1 dominates path2, using the last value entry of the paths.

        Args:
            path1 (list): Path one (one entry in a path looks like ((move_dir, shift_dir), value dict) ).
            path2 (list): Path two

        Returns:
            bool: Does path one dominate path 2
        """
        value_of_path1 = path1[-1][1] # Extracts value dict from current position in path
        value_of_path2 = path2[-1][1]

        a1, a2, a3 = value_of_path1["step_count"],value_of_path1["weight_shifted"],value_of_path1["distance_to_goal"]
        b1, b2, b3 = value_of_path2["step_count"], value_of_path2["weight_shifted"], value_of_path2["distance_to_goal"]

        return (
            (a1 <= b1 and a2 <= b2 and a3 <= b3) and
            (a1 < b1 or a2 < b2 or a3 < b3)
        )

    def update_pareto_paths(self, node: Node, path: list):
        """Updates the pareto front for a node.

        Args:
            node (Node): Node to be updated
            path (list): Path that is used for the update
        """

        dominated = [p for p in node._pareto_paths if self.path_domination(path, p)] # Get all paths from pareto paths that are dominated by the new one

        if not any(self.path_domination(p, path) for p in node._pareto_paths): # If there arent any paths in the pareto_paths that dominate the new path
            node._pareto_paths = [p for p in node._pareto_paths if p not in dominated]
            node._paths_changed = True
            node._pareto_paths.append(copy.deepcopy(path))
            
            # Then prune for if too many paths using epsilon domination
            #if len(node._pareto_paths) > self._max_solutions and node._depth != 0: # We dont prune at root since we want accurate pareto front there
            #    #print(f"Node at depth {node._depth} hat too many solutions, pruning ...")
            #    Helper.epsilon_clustering(node, max_archive_size=self._max_solutions)

    def backpropagate(self, node: Node, current_root: Node) -> None:
        """Backpropagate leaf metrics up the tree."""
        leaf_values = node._values.copy()
        path = []
        current = node
        current_root_parent = current_root._parent
        while current is not None and current is not current_root_parent:
            current._visits += 1

            # Incremental average update
            for key, val in leaf_values.items():
                current._values[key] = (
                    (current._values[key] * (current._visits - 1)) + val
                ) / current._visits
            move = current._last_move

            # Each node knows the move that lead to it
            # Save each move in path
            # Ignore child since it does not need to know its own origin move in path
            # Just append to path and go to parent
            # Add path that contains child move to parents pareto front
            # Add own origin move to path -> repeat till root

            if current is not node: # Last node does not need to even have a pareto path since its a leaf
                self.update_pareto_paths(current, path)
            
            if current._last_move is not None:
                path.append((move, current._values.copy()))

            current = current._parent

    def search(self, total_budget: int, per_sim_budget: int, simulations_per_child: int, rollout_func: int = 0, root_selection: int = 0, tree_selection: int = 0) -> None:
        """Methods that builds the tree and looks for solutions

        Args:
            total_budget (int): Total number of simulations that can be used for expansion in the whole tree.
            per_sim_budget (int): Maximum number of simulation steps per simulation.
            simulations_per_child (int): Number of rollouts per child from which best one is chosen.
            rollout_func (int): Indicator which rollout function to use.
            root_selection (int): Indicator which root selection function to use.
            tree_selection (int): Inidcator which tree selection function to use.
        """
        match root_selection:
            case 0: root_sel_function = Helper.epsilon_clustering_for_nodes
            case 1: root_sel_function = self.pareto_path_child_selection_hv
            case 2: root_sel_function = self.pareto_path_child_selection_cd
            case _: raise ValueError("Did not supply a suitable root selection indicator")
        
        match tree_selection:
            case 0: tree_sel_function = self.ucb_child_selection
            case 1: tree_sel_function = self.pareto_path_child_selection_hv
            case 2: tree_sel_function = self.pareto_path_child_selection_cd
            case _: raise ValueError("Did not supply a suitable tree selection indicator")
        
        match rollout_func:
            case 0: rollout_function = self.light_rollout
            case 1: rollout_function = self.iterative_heavy_angular_rollout
            case 2: rollout_function = self.iterative_heavy_distance_weight_rollout
            case _: raise ValueError("Did not supply a suitable rollout function indicator")

        log = Logger(self._root._controller._environment._map_type, self._root._controller._environment._env_dim, self._root._controller._start_pos, self._root._controller._environment._goal, total_budget, per_sim_budget, simulations_per_child, tree_sel_function.__name__, root_sel_function.__name__, self._max_solutions, rollout_function.__name__, self._seed, self._root)
        print("Starting search ...")
        # Make list for found solutions
            # Set root to initial root
        current_root = self._root
        solutions = []
        while not current_root.is_terminal_state():
            used_simulation_counter = 0
            while used_simulation_counter < total_budget:

                # Use tree policy
                # Reminder: Tree policy returns none if selected node has reached goal or we are in unsolvable state
                current_node = self.tree_policy(root = current_root, tree_sel_func=tree_sel_function)

                if current_node is not None:
                    # Expand the leaf (expand method automatically adds it to the current_nodes children and also returns it)
                    child = current_node.expand()

                    if child is not None: # expand returns none when we have no untried actions
                        self._max_depth = child._depth

                        if not child.is_terminal_state():
                            # Do simulations and add used budget onto sim counter
                            used_simulation_counter+=rollout_function(child, simulations_per_child, per_sim_budget, total_budget-used_simulation_counter)

                        self.backpropagate(child, current_root)
                    else:
                        used_simulation_counter += per_sim_budget # For fast convergence in the end

            # Current root umsetzen
            current_root = root_sel_function(current_root)
            #current_root = random.choice(Helper.determine_pareto_front_from_nodes(current_root._children.values()))
            #current_root = self.pareto_path_child_selection_hv(current_root)
            print(f"Root was set to {current_root._controller._current_pos}")
            if current_root.is_terminal_state():
                solutions.append(current_root)
        log.log_solutions(solutions)
