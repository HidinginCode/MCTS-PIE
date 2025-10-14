"""This module contains the class that creates and simulates the MCTS tree."""

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
            logging.StreamHandler()           # and also prints to console
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

    def select_node(self, current_node: Node) -> Node:
        """Method that traverses the tree untill leaf or an unexpanded node is found.

        Args:
            current_node (Node): Current node from where to start next selection

        Returns:
            Node: Selected Child node
        """

        depth = 0

        while current_node.get_children():
            current_node = current_node.select_child_pareto_ucb()
            depth +=1

        return current_node

    def expand(self, node_to_expand: Node) -> Node:
        """Expands the MCTS tree.

        Args:
            node_to_expand (Node): Node from which to expand

        Returns:
            Node: Random child node from expanded children
        """

        # If we have reached terminal state do not expand

        #########################################################
        if node_to_expand.get_state().get_terminal_state():
            return node_to_expand
        #########################################################

        node_to_expand.expand()
        valid_children = [
            child for child in node_to_expand.get_children().values() if child is not None
            ]
        for child in valid_children:
            if child.get_state().get_terminal_state():
                return child
        return random.choice(valid_children) if valid_children else node_to_expand

    def get_metrics(self, node: Node) -> dict:
        """Returns the metrics of the given node.

        Args:
            node (Node): Node for metric extraction

        Returns:
            dict: Metrics
        """

        return node.get_state().get_state_metrics()

    def backpropagate(self, current_node: Node) -> None:
        """Method that facilitates backpropagation to update precursor nodes.

        Args:
            current_node (Node): Current node from which update starts
        """

        logging.info("Starting backpropagation.")
        metrics = self.get_metrics(current_node)

        while current_node is not None:
            current_node.update_node(metrics)
            current_node = current_node.get_parent()

    def run_search(self, iterations: int = 1) -> None:
        """Runs the MCTS search for a specified number of iterations.

        Args:
            iterations (int, optional): Number of times to run MCTS. Defaults to 1.
        """
        for _ in range(iterations):
            leaf = self.select_node(self.root)
            if leaf.get_state().get_terminal_state():
                print("FOOKIN FOUND IT")
                print(str(leaf))
                input()
            expanded = self.expand(leaf)
            #if expanded.get_state().get_terminal_state():
            #    logging.info(f"Found terminal state. {str(expanded)}")
            #    break
            self.backpropagate(expanded)
