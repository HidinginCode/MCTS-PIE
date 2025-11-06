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


class MctsTree():
    """This class contains the mcts tree."""

    def __init__(self, root: Node):
        """Init method for the MCTS tree.

        Args:
            root (Node): Root node for the tree
        """
        self._identifier = id(self)
        self._root = root
        # Force the multiprocess start method to be fork for later memory sharing
        mp.set_start_method("fork", force=True)

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
                current_node = self.ucb_child_selection(current_node)


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

    def backpropagate(self, node: Node) -> None:
        """Backpropagates metrics from the new child node through the parents.
        This makes the parents metrics an average of the metrics of their children.

        Args:
            node (Node): Root of backpropagation
        """

        last_node = None
        current_node = node
        values_of_leaf = node._values
        path = []
        while current_node is not None:
            current_node._visits += 1
            # Check if current node has any children
            if current_node._children:
                # Iterate over each key/value
                for key in current_node._values:
                    value = 0
                    for child in current_node._children.values():
                        value += child._values[key]
                    value/=len(current_node._children)
                    current_node._values[key] = value

            # Append movement/shifting direction and values to path
            if last_node is not None:
                movement_pair = [key for key, value in current_node._children.items() if value is last_node][0]
                path.append((movement_pair, values_of_leaf))
                current_node._pareto_paths.append(list(path))

            if current_node._parent is None:
                print(f"\n\nRoot pareto paths: {current_node._pareto_paths[0]}")
            
            last_node = current_node
            current_node = current_node._parent

    def search(self, iterations: int) -> None:
        """Methods that builds the tree and looks for solutions

        Args:
            iterations (int): Number of search iterations to complete
        """

        # Make list for found solutions
        solutions = []

        for _ in tqdm(range(iterations)):

            # Use tree policy
            # Reminder: Tree policy returns none if selected node has reached goal or we are in unsolvable state
            current_node = self.tree_policy()

            if current_node is not None:
                # Expand the leaf (expand method automatically adds it to the current_nodes children and also returns it)
                child = current_node.expand()

                if child.is_terminal_state():
                    print("Found new solution")
                    solutions.append(child)
                else:
                    self.leaf_rollout(child, 16, 100, self.multiprocess_heavy_distance_rollout)
                    self.backpropagate(child)
        

        solutions = Helper.determine_pareto_front_from_nodes(solutions)
        for solution in solutions:
            solution: Node
            current = solution
            path = []
            while current is not None:
                path.append(current._controller._current_pos)
                current = current._parent
            path.reverse
            Analyzer.create_heatmap(solution._controller._environment._environment, solution._controller._start_pos, solution._controller._environment._goal, path)