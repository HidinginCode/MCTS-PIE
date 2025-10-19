"""This module contains the class that creates and simulates the MCTS tree."""

from copy import deepcopy
import logging
import random
#from multiprocessing import Pool
import os
import time
import pickle

from tqdm import tqdm
from node import Node

class McTree():
    """This class represents the MCTS tree."""

    # Configure the global logging behavior once
    logging.basicConfig(
        level=logging.INFO,  # or DEBUG for more verbosity
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )

    def __init__(self, root: Node):
        """Init method for McTree.

        Args:
            root (Node): Root node of tree
            max_depth (int): Max depth of tree
        """
        self.root = root
        self.max_depth = None
        self.identificator = id(self)

    def get_root(self) -> Node:
        """Returns root node of the tree.

        Returns:
            Node: Root node of the tree
        """

        return self.root

    def set_root(self, root: Node) -> None:
        """Sets new root for tree.

        Args:
            root (Node): New root node
        """

        self.root = root

    def get_max_depth(self) -> int:
        """Returns max depth of the tree.

        Returns:
            int: Max depth of tree
        """

        return self.max_depth

    def get_identificator(self) -> int:
        """Returns ID of the tree.

        Returns:
            int: ID of the tree
        """

        return self.identificator

    def select_node(self, root: Node) -> Node | None:
        """Slection method to get a leaf node or one that is not fully expanded.

        Args:
            root (Node): Root node

        Returns:
            Node | None: Leaf or not fully unexpanded node
        """

        # Check if node is fully expanded
        if not root.is_fully_expanded():
            return root

        # Check if root is terminal state (no expansion needed)
        # Return none so we dont have to check for terminal state in later steps
        if root.get_state().get_terminal_state():
            return None

        selected_node = self.select_node(self.child_selection(root))
        return selected_node

    def child_selection(self, node: Node) -> Node:
        """Selects a child based on the pareto front of children.

        Args:
            node (Node): Node from where child is selected.

        Returns:
            Node: Child node
        """

        # Check if front is available
        if len(node.get_front()) == 0:
            node.set_front(node.determine_pareto_front())

        if len(node.get_front()) == 0:
            print(f"Number of children: {len(node.get_children())}")
            input()

        front = node.get_front()
        visits = [child.get_visits() for child in front]


        if any(v == 0 for v in visits):
            probabilities = [1 / len(front)] * len(front)
        else:
            total_visits = sum(visits)
            probabilities = [v / total_visits for v in visits]

        return random.choices(front, probabilities, k=1)[0]
        #return random.choice(front)

    def expand(self, node: Node) -> Node:
        """Creates a child for a given node.

        Args:
            node (Node): Node to expand

        Returns:
            Node: New child node
        """

        # Get untried actions (all are valid)
        untried_actions = node.get_untried_actions()
        move_direction, shift_direction = random.choice(untried_actions)

        # Create new child and move it
        # No need to deepcopy state, it is deepcopied in childs constructor
        child = Node(node.get_state(), node)
        child.get_state().get_state_controller().move_agent(move_direction, shift_direction)
        child.values = child.get_state().get_state_metrics()
        child.parent_actions = (move_direction.name, shift_direction.name)
        node.children[(move_direction, shift_direction)] = child

        return child

    def simulate_leaf(self, leaf: Node, number_of_simulations: int, maximum_moves: int) -> None:
        """Function that simulates random rollouts from a leaf.

        Args:
            leaf (Node): Leaf node
            number_of_simulations (int): Number of simulations
            maximum_moves (int): Maximum number of moves per simulation
        """

        # Pre pickle leaf to avoid overhead
        # pickled_leaf = pickle.dumps(leaf)

        # if not leaf.get_state().get_terminal_state():
        #     # Obtain list of nodes from multiprocessing
        #     results = self.mult_pool.map(
        #         self.multiprocess_leaf_simulation,
        #         [(pickled_leaf, maximum_moves)] * number_of_simulations
        #     )

        results = self.iterative_leaf_simulation(leaf, number_of_simulations, maximum_moves)

        if results:
            non_dominated_results = leaf.determine_pareto_from_list(results)
            leaf.set_values(random.choice(non_dominated_results).get_values())

    @staticmethod
    def iterative_leaf_simulation(leaf: Node, number_of_sims: int, maximum_depth: int)->list[Node]:
        """Method that iteratively does the leaf simulations.

        Args:
            leaf (Node): Leaf for start of simulations
            number_of_sims (int): Number of simulations
            maximum_depth (int): Maximum number of views

        Returns:
            Node: Simulation solutions
        """

        solutions = []

        for _ in range(number_of_sims):
            leaf_copy = leaf.clone()
            copy_controller = leaf_copy.get_state().get_state_controller()
            for _ in range(maximum_depth):
                if leaf_copy.get_state().get_terminal_state():
                    break
                valid_moves = leaf_copy.get_all_valid_actions()
                move_direction, shift_direction= random.choice(valid_moves)
                copy_controller.move_agent(move_direction, shift_direction)

            solutions.append(leaf_copy)

        return solutions

    @staticmethod
    def multiprocess_leaf_simulation(args: tuple[Node, int]) -> Node:
        """Multiprocessing wrapper for leaf simulation.

        Args:
            args (any): Simulation Args

        Returns:
            Node: Copy of node after simulations
        """
        # Get random state for all workers
        random.seed(os.getpid() + time.time_ns())


        pickled_leaf, maximum_moves = args

        leaf = pickle.loads(pickled_leaf)

        # Copy leaf for independent states
        leaf_copy = deepcopy(leaf)
        copy_controller = leaf_copy.get_state().get_state_controller()

        for _ in range (maximum_moves):

            # Check if terminal state was reached
            if leaf_copy.get_state().get_terminal_state():
                break

            valid_moves = leaf_copy.get_all_valid_actions()
            move_direction, shift_direction= random.choice(valid_moves)
            copy_controller.move_agent(move_direction, shift_direction)

        return leaf_copy

    def backpropagate(self, node: Node) -> None:
        """Backpropagates metrics from the new child node through the parents.
        This makes the parents metrics an average of the metrics of their children.

        Args:
            node (Node): Root of backpropagation
        """

        current_node = node
        while current_node is not None:
            current_node.increase_visits(1)
            # Check if current node has any children
            if current_node.get_children():

                # Iterate over each key/value
                for key in current_node.get_values().keys():
                    value = 0
                    for child in current_node.get_children().values():
                        value += child.get_values()[key]
                    value/=len(current_node.get_children())
                    current_node.set_value(key, value)

            current_node = current_node.get_parent()

    def run_search(self, iterations: int = 1) -> None:
        """Runs the MCTS search for a specified number of iterations.

        Args:
            iterations (int, optional): Number of times to run MCTS. Defaults to 1.
        """

        solutions = []

        for _ in tqdm(range(iterations)):
            leaf = self.select_node(self.root)

            if leaf is not None:
                goal = leaf.get_state().get_state_controller().get_map_copy().get_goal()
                pos = leaf.get_state().get_state_controller().get_current_agent_position()

                if goal == pos and leaf not in solutions:
                    print(f"Found new solution at depth {leaf.get_depth()}")
                    solutions.append(leaf)
                self.simulate_leaf(leaf, 2, 500)
                new_child = self.expand(leaf)
                self.backpropagate(new_child)

        solutions = Node.determine_pareto_from_list(solutions)
        for solution in solutions:
            node = solution
            path = []
            actions = []
            while node is not None:
                path.append(node.get_state().get_state_controller().get_current_agent_position())
                actions.append(node.get_parent_actions())
                node = node.get_parent()
            print("####################################################################")
            path.reverse()
            actions.reverse()
            print(path)
            print(actions)
