"""This module represents a the node class for the MCTS tree."""

from __future__ import annotations
from copy import deepcopy
from directions import Direction
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
        self.visits = 0
        self.value = 0.0

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

    def get_visits(self) -> int:
        """Returns the number of times the node was visited.

        Returns:
            int: Number of visits
        """

        return self.visits

    def set_visits(self, visits: int) -> None:
        """Sets the number of node visits to a specific ammount.

        Args:
            visits (int): Number of times the node was visited.
        """

        self.visits = deepcopy(visits)

    def increase_visits(self, amount: int) -> None:
        """Increases the number of visits by a given ammount.

        Args:
            amount (int): Ammount of visits
        """

        self.visits += amount

    def get_value(self) -> float:
        """Returns the value of the node.

        Returns:
            float: Value of the node.
        """

        return self.value

    def set_value(self, value: float) -> None:
        """Sets the value to a specified amount.

        Args:
            value (float): New value
        """

        self.value = value

    def change_value(self, amount: float) -> None:
        """Changes the value by a given amount.

        Args:
            amount (float): Value change
        """
        self.value += amount

    def expand(self) -> None:
        """Expands the node, creating new children for all direction pairs 
        (movement_direction, shifting_direction)."""

        assert not self.children, (
            "Tried to expand on a node that already had children."
        )

        # We create 12 children, one for each movement and shiftind direction
        # (excluding the agents own position)
        for movement_direction in Direction:
            for shifting_direction in Direction:
                # We need the controller, the agent and the map to see if the action is valid
                # If it is we can create a leaf node, if not we put a None entry as placeholder
                copy_controller = deepcopy(self.state.get_state_controller())

                move_valid = copy_controller.move_agent(
                    move_direction=movement_direction,
                    shifting_direction=shifting_direction
                )

                if move_valid:
                    new_state = State(state_controller = copy_controller)
                    new_node = Node(state=new_state, parent=self)

                    self.children[(movement_direction, shifting_direction)] = new_node
                else:
                    self.children[(movement_direction, shifting_direction)] = None
