"""This module contains the class that creates and simulates the MCTS tree."""

from copy import deepcopy
from node import Node

class McTree():
    """This class represents the MCTS tree."""

    def __init__(self, root: Node, max_depth: int):
        """Init method for McTree.

        Args:
            root (Node): Root node of tree
            max_depth (int): Max depth of tree
        """

        self.root = deepcopy(root)
        self.max_depth = deepcopy(max_depth)
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

        self.root = deepcopy(root)

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
