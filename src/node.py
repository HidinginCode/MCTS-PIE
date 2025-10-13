"""This module represents a the node class for the MCTS tree."""

from __future__ import annotations
from copy import deepcopy
from state import State

class Node():
    """This is the node class, which is used for mcts."""

    def __init__(self, state: State, parent: Node | None = None):
        """Init method for the node class.

        Args:
            state (State): State in the current node.
            parent (Node | None, optional): Parent node. Defaults to None.
        """

        self.state = deepcopy(state)
        self.parent = deepcopy(parent)
        self.children = {}
        self.identificator = id(self)

    def get_state(self) -> State:
        """Retruns the state of the current node.

        Returns:
            State: State of the current node
        """

        return self.state

    def get_parent(self) -> Node | None:
        """Returns the parent node of the current node.

        Returns:
            Node: Parent node
        """

        return self.parent

    def set_parent(self, parent: Node | None) -> None:
        """Sets the parent of the current node.

        Args:
            parent (Node | None): New parent for node
        """
        self.parent = deepcopy(parent)

    def get_children(self) -> dict | None :
        """Returns children of the current Node.

        Returns:
            dict | None: Children of current node
        """

        return self.children

    def set_children(self, children: dict) -> None:
        """Sets children of current node.

        Args:
            children (dict): New children of current node
        """

        self.children = deepcopy(children)

    def get_identificator(self) -> int:
        """Returns ID of current node.

        Returns:
            int: ID of node
        """

        return self.identificator
