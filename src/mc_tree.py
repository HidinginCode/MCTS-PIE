"""This module contains the class that creates and simulates the MCTS tree."""

from copy import deepcopy
import logging
import random

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

    def __init__(self, root: Node, max_depth: int):
        """Init method for McTree.

        Args:
            root (Node): Root node of tree
            max_depth (int): Max depth of tree
        """

        self.root = root
        self.max_depth = max_depth
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

        return random.choice(node.get_front())

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
        child = Node(deepcopy(node.get_state()), node)
        child.get_state().get_state_controller().move_agent(move_direction, shift_direction)
        child.values = child.get_state().get_state_metrics()
        node.children[(move_direction, shift_direction)] = child

        return child

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
        for _ in range(iterations):
            leaf = self.select_node(self.root)

            if leaf is not None:
                new_child = self.expand(leaf)
                self.backpropagate(new_child)
