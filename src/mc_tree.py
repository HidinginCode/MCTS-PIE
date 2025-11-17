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
from tqdm import tqdm
from analyzer import Analyzer
import copy


class MctsTree():
    """This class contains the mcts tree."""

    def __init__(self, root: Node, max_solutions: int = 10):
        """Init method for the MCTS tree.

        Args:
            root (Node): Root node for the tree
            max_solutions (int): Maximum number of entries in pareto front
        """
        self._identifier = id(self)
        self._root = root
        self._max_solutions = max_solutions
        # Force the multiprocess start method to be fork for later memory sharing
        mp.set_start_method("fork", force=True)
        self._max_depth = 0

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

    def tree_policy(self) -> Node | None:
        """Tree policy that selects the path from the root to a leaf.

        Returns:
            Node | None: Either leaf node or none if i.e. solution was selected.
        """

        current_node = self._root
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
                current_node = self.pareto_path_child_selection_hv(current_node)

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
        """Selects children based on pareto dominance of UCB1-Calculations

        Args:
            node (Node): Node of which children are selected

        Returns:
            Node: Child node
        TODO: Figure out other selection than random choice (Hypervolume, Crowding-Distance,...)
        """
        children = node._children
        number_of_children = len(children)

        for child in children.values():
            child: Node
            dimensions = len(child._values) # Number of dimensions
            child_visits = child._visits
            parent_visits = node._visits
            exploration_term = np.sqrt(
                (2*np.log(
                    parent_visits*np.sqrt(np.sqrt(dimensions*number_of_children)))
                )/child_visits
            )
            child._ucb_values = {k: v - exploration_term for k, v in child._values.items()}

        pareto_front = Helper.determine_pareto_front_from_nodes(children.values(), True)
        return random.choice(pareto_front)

    def iterative_heavy_distance_rollout(self, leaf: Node, simulations: int, maximum_moves: int):
        """Iterative version of the heavy distance rollout to look into performance.

        Args:
            leaf (Node): Leaf to simulate
            simulations (int): Number of iterative simulations
            maximum_moves (int): Maximum number of moves per simulation
        """
        results = []
        manhattan = lambda p, q: abs(p[0]-q[0]) + abs(p[1]-q[1])
        for _ in range(simulations):
            leaf_copy = leaf.clone()
            controller = leaf_copy._controller
            goal = controller._environment._goal

            for _ in range(maximum_moves):
                # Break if we reached terminal state
                if leaf_copy.is_terminal_state():
                    break

                current_pos = controller._current_pos
                # Get needed parts of calculation and prepare move list
                current_distance_to_goal = manhattan(current_pos, goal)
                distance_minimizing_moves = []
                valid_moves = leaf_copy._controller.get_all_valid_pairs()
                # Get moves that do not increase distance
                for move_dir, shifting_dir in valid_moves:
                    new_pos = (current_pos[0] + move_dir.value[0],
                            current_pos[1] + move_dir.value[1])
                    new_distance_to_goal = manhattan(new_pos, goal)
                    if new_distance_to_goal <= current_distance_to_goal:
                        distance_minimizing_moves.append((move_dir, shifting_dir))

                # Randomly chose from moves
                move_direction, shift_direction = random.choice(distance_minimizing_moves)
                controller.move(move_direction, shift_direction)
            results.append(leaf_copy.clone())
        
        chosen_node = random.choice(Helper.determine_pareto_front_from_nodes(results))
        leaf._values = dict(chosen_node._values)

    def leaf_rollout(self, leaf: Node, simulations: int, maximum_moves: int, rollout_method: function) -> None:
        """Rollout method for a given leaf. 
        Acts more as a wrapper for the real multiprocessing rollouts.

        Args:
            leaf (Node): Leaf to be rolled out
            simulations (int): Number of simultaneous simulations
            maximum_moves (int): Maximum number of moves per simulation
            rollout_method (function): Rollout function to be used
        """

        
        MctsTree.SHARED_NODE = leaf
        number_of_processes = min(os.cpu_count(), simulations)
        with mp.Pool(processes=number_of_processes) as p:
            it = p.imap_unordered(rollout_method, [maximum_moves]*number_of_processes)
            results = list(it)
        
        if results:
            chosen_node = random.choice(Helper.determine_pareto_front_from_nodes(results))
            leaf._values = dict(chosen_node._values)
        else:
            raise RuntimeError("Leaf rollout returned no results")

    @staticmethod
    def multiprocess_heavy_distance_rollout(maximum_moves: int) -> Node:
        """Heavy rollout for leaf simulation.
        Moves are selected by either decreasing or staying at the same distance to the goal.

        Args:
            maximum_moves (int): Number of maximum moves till break

        Returns:leaf_copy = Node(McTree.SHARED_NODE.state.clone(), None)
            Node: Simulated node
        """

        leaf_copy = MctsTree.SHARED_NODE.clone()
        leaf_copy._parent = None
        controller = leaf_copy._controller
        goal = controller._environment._goal
        manhattan = lambda p, q: abs(p[0]-q[0]) + abs(p[1]-q[1])

        for _ in range(maximum_moves):
            # Break if we reached terminal state
            if leaf_copy.is_terminal_state():
                break

            current_pos = controller._current_pos
            # Get needed parts of calculation and prepare move list
            current_distance_to_goal = manhattan(current_pos, goal)
            distance_minimizing_moves = []
            valid_moves = leaf_copy._controller.get_all_valid_pairs()
            # Get moves that do not increase distance
            for move_dir, shifting_dir in valid_moves:
                new_pos = (current_pos[0] + move_dir.value[0],
                           current_pos[1] + move_dir.value[1])
                new_distance_to_goal = manhattan(new_pos, goal)
                if new_distance_to_goal <= current_distance_to_goal:
                    distance_minimizing_moves.append((move_dir, shifting_dir))

            # Randomly chose from moves
            move_direction, shift_direction = random.choice(distance_minimizing_moves)
            controller.move(move_direction, shift_direction)
        return leaf_copy

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

        a1, a2, a3 =value_of_path1["step_count"],value_of_path1["weight_shifted"],value_of_path1["distance_to_goal"]
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
            if len(node._pareto_paths) > self._max_solutions and node._depth != 0: # We dont prune at root since we want accurate pareto front there
                #print(f"Node at depth {node._depth} hat too many solutions, pruning ...")
                Helper.epsilon_clustering(node, max_archive_size=self._max_solutions)

    def backpropagate(self, node: Node) -> None:
        """Backpropagate leaf metrics up the tree."""
        leaf_values = node._values.copy()
        path = []
        current = node

        while current is not None:
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
                path.append((move, leaf_values.copy()))

            current = current._parent

    def search(self, iterations: int) -> None:
        """Methods that builds the tree and looks for solutions

        Args:
            iterations (int): Number of search iterations to complete
        """

        print("Starting search ...")
        # Make list for found solutions
        solutions = []

        for _ in tqdm(range(iterations)):

            # Use tree policy
            # Reminder: Tree policy returns none if selected node has reached goal or we are in unsolvable state
            current_node = self.tree_policy()

            if current_node is not None:
                # Expand the leaf (expand method automatically adds it to the current_nodes children and also returns it)
                child = current_node.expand()
                self._max_depth = child._depth

                if child.is_terminal_state():
                    if child not in solutions:
                        print(f"Found new solution at depth: {child._depth}")
                        solutions.append(child)
                    else:
                        print(f"Found already known solution at depth {child._depth}")
                    continue
                else:
                    self.iterative_heavy_distance_rollout(child, 16, 500)
                    self.backpropagate(child)

        for solution in solutions:
            solution.refresh_values()
        
        print(f"Solutions before pruning: {len(solutions)}")
        solutions = Helper.determine_pareto_front_from_nodes(solutions)
        print(f"Solutions after pruning: {len(solutions)}")
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
                    shifts.append(current._last_move[1].value)
                current = current._parent
            
            moves.reverse()
            shifts.reverse()
            path.reverse()

            Analyzer.visualize_path_with_shifts(solution._controller._environment._environment, path, shifts, (0,0), solution._controller._environment._goal, f"./log/heatmap-{i}.png")
            Analyzer.save_path_as_gif(self.root._controller._environment, self.root._controller._start_pos, moves, f"./log/heatmap-{i}.gif")
            #Analyzer.interactive_step_path(self.root._controller._environment, self.root._controller._start_pos, moves)
        # if iterations < 20000:
        #     Analyzer.visualize_mcts_svg(self._root, "./log/tree.svg")