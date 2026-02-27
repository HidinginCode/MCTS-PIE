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
    
    def _can_expand(self, node: Node, C: float = 1.5, alpha: float = 0.5) -> bool:
            """Progressive widening rule."""
            num_children = len(node._children)
            max_children = C * (node._visits ** alpha)
            return num_children < max_children

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
                #print("Found terminal state")
                return None

            # Check if the node is fully expanded
            if current_node.get_untried_actions() and self._can_expand(current_node):
                #print("Found node with untried actions")
                return current_node

            # At this point we know that current_node is neither terminal nor has any expansion left
            # Safety check for children
            if current_node._children:
                #print(f"Using tree policy: {tree_sel_func.__name__}")
                current_node = tree_sel_func(current_node)

    def hv_root_selection(self, root: Node) -> Node:
        """Selects new root based on hypervolume of children.
        Children of current root are passed through pareto filter and then HV contribution is calculated.

        Args:
            root (Node): Root node from which to choose one child

        Returns:
            Node: Chosen child node.
        """
        pareto_children = Helper.determine_pareto_front_from_nodes(root._children.values())
        if not pareto_children:
            return random.choice(list(root._children.values()))
        
        if len(pareto_children) == 1:
            return pareto_children[0]
        
        values = [child._values for child in pareto_children]
        hv_contrib = np.asarray(Helper.hypervolume_contributions(values), dtype=float)

        if np.sum(hv_contrib) <= 1e-12:
            return random.choice(pareto_children)
        
        probs = hv_contrib / np.sum(hv_contrib)
        child_index = np.random.choice(len(pareto_children), p=probs)

        return pareto_children[child_index]

    def pareto_path_child_selection_aec(self, node: Node) -> Node:
        """
        Strong adaptive epsilon clustering child selection.

        Improvements over basic AEC:
        - adaptive number of clusters based on visits
        - normalized objective space
        - quality-weighted sampling
        - exploration bonus (UCB-style)
        - safe fallbacks
        """

        children = list(node._children.values())
        if not children:
            raise RuntimeError("AEC selection called on node without children")

        # ---------- fallback if no pareto info ----------
        if not node._pareto_paths:
            return random.choice(children)

        # ---------- adaptive number of clusters ----------
        # grows slowly with experience
        number_of_paths = max(2, int(np.sqrt(node._visits)))

        # ---------- cluster pareto paths ----------
        rep_paths = Helper.adaptive_epsilon_archiving_selection(node=node, desired_p_number=number_of_paths)

        if not rep_paths:
            return random.choice(children)

        # ---------- extract objective values ----------
        value_dicts = [path[-1][1] for path in rep_paths]

        # normalize objectives for fair comparison
        norm_values = Helper.normalize_archive(value_dicts)

        # ---------- compute sampling weights ----------
        weights = []

        parent_visits = max(1, node._visits)
        logN = np.log(parent_visits)

        exploration_strength = 0.5   # tune if needed

        for path, norm_dict in zip(rep_paths, norm_values):

            # --- quality term (minimization) ---
            vec = np.array(list(norm_dict.values()), dtype=float)
            quality = 1.0 / (np.linalg.norm(vec) + 1e-9)

            # --- exploration bonus ---
            child_key = path[-1][0]
            child = node._children.get(child_key)

            if child is None or child._visits == 0:
                exploration = 1.0   # strong push to unexplored
            else:
                exploration = exploration_strength * np.sqrt(logN / child._visits)

            weights.append(quality + exploration)

        # guard against all-zero weights
        if sum(weights) <= 0:
            return random.choice(children)

        # ---------- sample representative ----------
        chosen_path = random.choices(rep_paths, weights=weights, k=1)[0]

        # ---------- follow corresponding child ----------
        child_key = chosen_path[-1][0]
        return node._children.get(child_key, random.choice(children))

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
            extreme_mass = 0.2
            finite_mass = 0.8

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
            values = [path[-1][1] for path in node._pareto_paths]
            contrib = Helper.hypervolume_contributions(values)
            # Since paths changed we save new hypervolumes
            node._old_hv_values = contrib
            node._paths_changed = False
        else:
            contrib = node._old_hv_values

        child_key_index = random.choices(range(len(contrib)), weights=contrib, k=1)[0]
        child_key = node._pareto_paths[child_key_index][-1][0]
        return node._children[child_key]

    def ucb_child_selection(self, node: Node) -> Node:
        """Select children based on Pareto dominance of normalized UCB1 values (minimization)."""
        children = list(node._children.values())
        if not children:
            raise RuntimeError("ucb_child_selection called on node without children")

        dims = list(children[0]._values.keys())

        # Use robust z-score normalization across children
        raw_values = [child._values for child in children]
        normalized_list = Helper.stable_normalize(raw_values)

        parent_visits = max(1, node._visits)
        logN = np.log(parent_visits)

        # multi-objective exploration weights (keep your original alphas)
        alpha = {
            "distance_to_goal": 0.3,
            "step_count": 1.0,
            "weight_shifted": 1.0,
        }

        for child, norm_vals in zip(children, normalized_list):
            # Force exploration of unvisited children
            if child._visits == 0:
                child._ucb_values = {d: -np.inf for d in dims}
                continue

            for d in dims:
                # norm_vals[d] is already in a balanced z-score scale
                # Minimization: lower is better
                exploit = norm_vals[d]
                explore = alpha[d] * np.sqrt(2 * logN / child._visits)
                child._ucb_values[d] = exploit - explore

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


    def iterative_heavy_square_sampling_rollout(self, leaf: Node,
                                        simulations: int,
                                        maximum_moves: int,
                                        remaining_budget: int) -> int:
        """
        Iterative rollout method that:
        1) Samples points in a square radius around the current position
        2) Chooses a sampled point that is closest (Manhattan) to the goal position
        3) Moves greedily towards that point, preferring low-weight cells
        4) Uses the controller's (move_dir, shift_dir) pairs to actually move

        Objectives behind the heuristic:
        - Fewer steps (shorter geometric paths)
        - Less total weight shifted (avoid heavy cells where possible)
        """

        sampling_radius = max(2, env_size // 10)
        sample_count = 5
        used_moves_total = 0

        results: list[Node] = []

        def manhattan_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
            ax, ay = a
            bx, by = b
            dx = ax - bx
            dy = ay - by
            return (dx if dx >= 0 else -dx) + (dy if dy >= 0 else -dy)

        # 4-neighborhood movement
        CARDINAL_DIRECTIONS = [
            (1, 0),   # east
            (-1, 0),  # west
            (0, 1),   # south
            (0, -1)   # north
        ]

        def greedy_step_towards_point(start_pos: tuple[int, int],
                                      target_pos: tuple[int, int],
                                      controller: Controller) -> list[tuple[int, int]]:
            """
            Greedy walk from start_pos towards target_pos.

            - Only uses cardinal moves.
            - Among moves that improve Manhattan distance to target_pos,
              it prefers those with minimal cell weight at the destination.
            - Returns a list of (dx, dy) deltas, NOT absolute positions.
            """

            env = controller._environment
            env_grid = env._environment          # 2D weight map
            env_dim = env.env_dim

            current_x, current_y = start_pos
            target_x, target_y = target_pos

            chosen_deltas: list[tuple[int, int]] = []

            while (current_x, current_y) != (target_x, target_y):

                current_dist = manhattan_distance(
                    (current_x, current_y),
                    (target_x, target_y)
                )

                # First filter: directions that strictly improve Manhattan distance
                improving_dirs: list[tuple[int, int]] = []
                best_dist = current_dist

                for dx_move, dy_move in CARDINAL_DIRECTIONS:
                    next_x = current_x + dx_move
                    next_y = current_y + dy_move

                    # Bounds check
                    if not (0 <= next_x < env_dim and 0 <= next_y < env_dim):
                        continue

                    new_dist = manhattan_distance(
                        (next_x, next_y),
                        (target_x, target_y)
                    )

                    if new_dist < best_dist:
                        best_dist = new_dist
                        improving_dirs = [(dx_move, dy_move)]
                    elif new_dist == best_dist:
                        improving_dirs.append((dx_move, dy_move))

                # No improving direction -> local minimum w.r.t. this target -> stop
                if not improving_dirs:
                    break
                
                # Random tie-break among equally good (distance, weight) directions
                dx_chosen, dy_chosen = random.choice(improving_dirs)

                chosen_deltas.append((dx_chosen, dy_chosen))
                current_x += dx_chosen
                current_y += dy_chosen

            return chosen_deltas

        # ----- MAIN ROLLOUT LOOP -----
        for _ in range(simulations):

            leaf_clone = leaf.clone()
            controller = leaf_clone._controller

            environment = controller._environment
            env_size = environment.env_dim
            

            for _ in range(maximum_moves):

                # Stop if terminal or out of global budget
                if leaf_clone.is_terminal_state() or used_moves_total >= remaining_budget:
                    break

                current_x, current_y = controller._current_pos
                if not controller._goal_collected:
                    goal_position = environment._goal
                else:
                    goal_position = environment._start_pos

                # Sampling square bounds
                min_x = max(0, current_x - sampling_radius)
                max_x = min(env_size - 1, current_x + sampling_radius)
                min_y = max(0, current_y - sampling_radius)
                max_y = min(env_size - 1, current_y + sampling_radius)

                goal_x, goal_y = goal_position

                # If the true goal lies inside our local sampling window, target it
                if min_x <= goal_x <= max_x and min_y <= goal_y <= max_y:
                    next_sample_point = goal_position
                else:
                    # Otherwise: sample a few points around us and pick the closest to the goal
                    next_sample_point = min(((random.randint(min_x, max_x), random.randint(min_y, max_y)) for _ in range(sample_count)), key=lambda p: manhattan_distance((goal_x, goal_y), p))

                # Get direction deltas for greedy movement towards next_sample_point
                direction_deltas = greedy_step_towards_point((current_x, current_y), next_sample_point, controller)

                # Apply those deltas as actual (move_dir, shift_dir) pairs
                for delta_dx, delta_dy in direction_deltas:
                    if used_moves_total >= remaining_budget:
                        break

                    valid_pairs = [pair for pair in controller.get_all_valid_pairs() if pair[0] == (delta_dx, delta_dy)]

                    # If no valid (move,shift) exists for that move_dir, stop following this path
                    if not valid_pairs:
                        break

                    move_dir, shift_dir = random.choice(valid_pairs)
                    controller.move(move_dir, shift_dir)
                    used_moves_total += 1

            results.append(leaf_clone)

        chosen_leaf = random.choice(Helper.determine_pareto_front_from_nodes(results))
        leaf._values = dict(chosen_leaf._values)

        return used_moves_total


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
        
        if len(node._pareto_paths) > self._max_solutions:
            Helper.epsilon_clustering(node=node, max_archive_size=self._max_solutions)
            

    def backpropagate(self, node: Node, current_root: Node) -> None:
        """Backpropagate leaf metrics up the tree."""

        # Use stable, unmodified totals (not the running mean)
        leaf_values = node._real_values.copy()

        path = []
        current = node
        current_root_parent = current_root._parent

        while current is not None and current is not current_root_parent:
            current._visits += 1

            # Keep your averaging exactly as-is, but average leaf_values (stable)
            for key, val in leaf_values.items():
                current._values[key] = (
                    (current._values[key] * (current._visits - 1)) + val
                ) / current._visits

            move = current._last_move
            if current is not node:
                self.update_pareto_paths(current, path)

            if move is not None:
                # Build a node-relative ("to-go") objective vector for the archive
                vals = leaf_values.copy()

                # This to go approach makes nodes comparable at different depths of the tree objective wise
                vals["step_count"] = leaf_values["step_count"] - current._real_values["step_count"]
                vals["weight_shifted"] = leaf_values["weight_shifted"] - current._real_values["weight_shifted"]
                vals["distance_to_goal"] = leaf_values["distance_to_goal"]  # simplest/minimal disruption

                path.append((move, vals))

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
            case 0: root_sel_function = self.hv_root_selection
            case _: raise ValueError("Did not supply a suitable root selection indicator")
        
        match tree_selection:
            case 0: tree_sel_function = self.ucb_child_selection
            case 1: tree_sel_function = self.pareto_path_child_selection_hv
            case 2: tree_sel_function = self.pareto_path_child_selection_cd
            case 3: tree_sel_function = self.pareto_path_child_selection_aec
            case _: raise ValueError("Did not supply a suitable tree selection indicator")
        
        match rollout_func:
            case 0: rollout_function = self.light_rollout
            case 1: rollout_function = self.iterative_heavy_square_sampling_rollout
            case 2: rollout_function = self.iterative_heavy_distance_weight_rollout
            case _: raise ValueError("Did not supply a suitable rollout function indicator")

        log = Logger(self._root._controller._environment._map_type, self._root._controller._environment._env_dim, self._root._controller._start_pos, self._root._controller._environment._goal, total_budget, per_sim_budget, simulations_per_child, tree_sel_function.__name__, root_sel_function.__name__, self._max_solutions, rollout_function.__name__, self._seed, self._root)
        print("Starting search ...")
        # Make list for found solutions
            # Set root to initial root
        current_root = self._root
        solutions = []
        while (not current_root.is_terminal_state())and current_root._depth <= 400:
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
                else:
                        used_simulation_counter += per_sim_budget

            # Current root umsetzen
            current_root = root_sel_function(current_root)
            #print(f"New root at depth: {current_root._depth}")
            Node.prune_siblings(current_root) # Remove siblings to prune tree

            if current_root.is_terminal_state():
                solutions.append(current_root)
        log.log_solutions(solutions)
