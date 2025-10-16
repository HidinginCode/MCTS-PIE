"""This module contains the class that creates and simulates the MCTS tree."""

import logging
import random
import tqdm

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

    def select_node(self, current_node: Node) -> Node:
        """Method that traverses the tree untill leaf or an unexpanded node is found.

        Args:
            current_node (Node): Current node from where to start next selection

        Returns:
            Node: Selected Child node
        """

        while True:
            if current_node.get_state().get_terminal_state():
                return current_node
            if not self.is_fully_expanded(current_node):
                return current_node

            candidates = [
                child for child in current_node.get_children().values() if child is not None
            ]
            if not candidates:
                return current_node

            current_node = current_node.select_child_pareto_ucb()

    def is_fully_expanded(self, node: Node) -> bool:
        """Checks if a node is fully expanded already.

        Args:
            node (Node): Node to check

        Returns:
            bool: Is fully expanded or not
        """
        all_valid_directions = node.get_all_valid_directions()
        return set(node.get_children().keys()) == set(all_valid_directions)

    def expand(self, node_to_expand: Node) -> Node:
        """Expands the MCTS tree.

        Args:
            node_to_expand (Node): Node from which to expand

        Returns:
            Node: Random child node from expanded children
        """

        if node_to_expand.get_state().get_terminal_state():
            return node_to_expand
        new_child = node_to_expand.expand()

        if new_child is not None:
            return new_child

        # Just keep search moving
        valid_children = [
            child for child in node_to_expand.get_children().values()
            if child is not None
        ]
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

        #logging.info("Starting backpropagation at depth %s", current_node.get_depth())

        node = current_node

        while node is not None:
            node.increase_visits(1)
            children = [
                child for child in node.get_children().values() if child is not None
            ]

            if not children:
                # If we are at a leaf we keep the averaged values of the simulation
                metrics = node.get_values()
                # Make sure we got front
                if not node.get_front():
                    node.set_front([metrics])
                all_vectors = node.get_front()

            else:
                # If we have children we compute the average of the childrens metrics
                metrics = {
                    key: sum(child.get_values()[key] for child in children) /len(children)
                    for key in node.get_values()
                }

                # Collect pareto fronts from children and remove dominated solutions
                all_vectors = []
                for child in children:
                    if child.get_front():
                        all_vectors.extend(child.get_front())
                    else:
                        all_vectors.append(child.get_values())

            node.set_front(node.pareto_reduction(all_vectors))

            # Update value with chosen metric
            node.set_value(metrics)
            node = node.get_parent()

    def run_search(self, iterations: int = 1) -> None:
        """Runs the MCTS search for a specified number of iterations.

        Args:
            iterations (int, optional): Number of times to run MCTS. Defaults to 1.
        """
        for _ in tqdm.tqdm(range(iterations)):
            leaf = self.select_node(self.root)


            #if leaf.get_state().get_terminal_state():
            #    print("FOOKIN FOUND IT")
            #    print(str(leaf))
            #    input()

            expanded = self.expand(leaf)
            #if expanded.get_state().get_terminal_state():
            #    logging.info(f"Found terminal state. {str(expanded)}")
            #    break
            expanded.simulate_leaf(expanded, maximum_depth=3, number_of_simulations=4)
            self.backpropagate(expanded)
