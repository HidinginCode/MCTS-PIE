"""This module contains the node class which later builds the MCTS tree."""
from __future__ import annotations
from controller import Controller
import random

class Node():
    """This node class is the basis for later building the MCTS tree."""

    def __init__(self, controller: Controller, parent: Node | None = None):
        """Init method for the node class that accepts a controller and a parent node.

        Args:
            controller (Controller): Controller for the node
            parent (Node | None, optional): Possible parent of the node. Defaults to None.
        """

        self._identifier = id(self)
        self._controller = controller.clone()
        self._parent = parent
        if parent is not None:
            self._depth = int(self._parent._depth) + 1
        else:
            self._depth = 0
        self._children = {}
        self._visits = 1
        self._values = {"step_count": self._controller._step_count,
                        "weight_shifted": self._controller._weight_shifted,
                        "distance_to_goal": self._controller._distance_to_goal}
        self._ucb_values = {"step_count": 0,
                            "weight_shifted": 0,
                            "distance_to_goal": 0
                            }
        self._pareto_paths = [] # This holds action, movement and values like the following ((move_dir, action_dir), value_dict)

    def clone(self) -> Node:
        """Clone method for node object.

        Returns:
            Node: Cloned node
        """
        new_node = Node(self._controller, self._parent)
        new_node._depth = int(self._depth)
        new_node._children = self._children
        new_node._visits = int(self._visits)
        new_node._values = dict(self._values)
        new_node._ucb_values = dict(self._ucb_values)
        new_node._pareto_paths = list(self._pareto_paths)

        return new_node

    @property
    def identifier(self) -> int:
        """Getter for identifier.

        Returns:
            int: ID of node
        """
        return self._identifier

    @property
    def controller(self) -> int:
        """Getter for controller.

        Returns:
            int: Controller of node
        """
        return self._controller

    @property
    def parent(self) -> int:
        """Getter for parent.

        Returns:
            node: parent of node
        """
        return self._parent
    
    @parent.setter
    def parent(self, new_parent: Node) -> None:
        """Setter for parent of node

        Args:
            new_parent (Node): New parent of node.
        """
        self._parent = new_parent
        self._depth = new_parent._depth + 1

    @property
    def children(self) -> dict:
        """Getter for children of node

        Returns:
            dict: Children of node.
        """
        return self._children

    @property
    def visits(self) -> int:
        """Getter for visits of node.

        Returns:
            int: Visits of node
        """
        return self._visits

    @visits.setter
    def visits(self, new_visits: int) -> None:
        """Visit setter for nodes.

        Args:
            new_visits (int): New visits for node.
        """
        self._visits = new_visits

    @property
    def values(self) -> dict:
        """Getter for values of node.

        Returns:
            dict: Values of node
        """
        return self._values

    def refresh_values(self) -> None:
        """Retrieves the current values from the controller and loads them into the dict."""
        self._values = {"step_count": self._controller._step_count,
                        "weight_shifted": self._controller._weight_shifted,
                        "distance_to_goal": self._controller._distance_to_goal}

    def get_untried_actions(self) -> list[tuple]:
        """Returns all actions that were not yet tried on that node.

        Returns:
            list[tuple]: Untried actions
        """
        valid_actions = self._controller.get_all_valid_pairs()
        untried_actions = [action for action in valid_actions if action not in self._children]
        return untried_actions

    def is_terminal_state(self) -> bool:
        """Method that checks if the current state is terminal

        Returns:
            bool: Is terminal
        """
        return self._controller._distance_to_goal == 0
    
    def expand(self) -> Node:
        """Method that adds a new child to the current node if there are untried actions.

        Returns:
            Node: New child node
        """

        if self.get_untried_actions():
            move_pair = random.choice(self.get_untried_actions())
            move_dir, shift_dir = move_pair
            child_node = Node(self._controller, self)
            child_node._controller.move(move_dir, shift_dir)
            # After move we load the new objective values into value dict
            child_node.refresh_values()
            self._children[move_pair] = child_node
            return child_node
        else:
            raise RuntimeError("Attempted to expand a fully expanded node.")