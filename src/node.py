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

        self.depth = parent.depth + 1 if parent else 0
        self.state = deepcopy(state)
        self.parent = parent
        self.children = {}
        self.identificator = id(self)
        self.visits = 0
        self.values = {
            "step_count": 0,
            "weight_shifted": 0.0,
            "distance_to_goal": 0,
        }
        self.ucb_vector = {}
        self.front = []

    def __str__(self):
        """String method for the node class."""
        return f"Visits: {self.visits}\nValues: {self.values}\nID: {self.identificator}"

    def get_depth(self) -> int:
        """Returns the depth of a node.

        Returns:
            int: Depth of node
        """

        return self.depth

    def set_depth(self, depth: int) -> None:
        """Sets a specified depth for the node.

        Args:
            depth (int): depth
        """

        self.depth = depth

    def get_state(self) -> State:
        """Returns the state of the current node.

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
        self.parent = parent

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

        self.children = children

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
        """Sets the number of node visits to a specific amount.

        Args:
            visits (int): Number of times the node was visited.
        """

        self.visits = visits

    def increase_visits(self, amount: int) -> None:
        """Increases the number of visits by a given amount.

        Args:
            amount (int): Amount of visits
        """

        self.visits += amount

    def get_values(self) -> dict:
        """Returns the value of the node.

        Returns:
            float: Value of the node.
        """

        return self.values

    def set_value(self, key: any, value: int | float) -> None:
        """Sets the value to a specified amount.

        Args:
            key: Key in values dictionary
            value (float): New value
        """

        self.values[key] = value

    def get_front(self) -> list:
        """Returns the pareto front of all children.

        Returns:
            list: Pareto front of children
        """
        return self.front

    def set_front(self, front: list) -> None:
        """Sets the pareto front for this node.

        Args:
            front (list): New pareto front.
        """
        self.front = front

    def get_all_valid_actions(self) -> list:
        """Returns all valid actions in this state.

        Returns:
            list: List of movement and shfiting direction tuples.
        """

        valid_action_pairs = []

        controller = self.get_state().get_state_controller()
        for move_direction in Direction:

            # See if direction is valid
            move_valid = controller.is_valid_direction(
                            move_direction,
                            controller.get_current_agent_position()
                        )

            if move_valid:
                for shift_direction in Direction:

                    new_pos = tuple(
                        sum(coord) for coord in zip(
                                controller.get_current_agent_position(),
                                move_direction.value)
                    )

                    # See if shift is valid from new pos
                    shift_valid = controller.is_valid_direction(
                        shift_direction,
                        new_pos
                    )

                    if shift_valid:
                        valid_action_pairs.append((move_direction, shift_direction))

        return valid_action_pairs

    def get_untried_actions(self) -> list:
        """Returns valid actions that are not yet expanded on.

        Returns:
            list: New valid actions
        """
        all_valid_actions = self.get_all_valid_actions()
        expanded_actions = [child for child in self.children.keys()]

        untried_actions = [pair for pair in all_valid_actions if pair not in expanded_actions]

        return untried_actions

    def is_fully_expanded(self) -> bool:
        """Method that determines if a node is fully expanded."""

        untried_actions = self.get_untried_actions()
        if len(untried_actions) == 0:
            return True
        return False

    def is_dominated(self, node: Node) -> bool:
        """Determines if the current node is dominated by the specified one.

        Args:
            node (Node): Node to be checked for domination.

        Returns:
            bool: Is dominated or not
        """
        node_values = node.get_values()
        return(
            all(node_values[key] <= self.values[key] for key in node_values.keys()) and
            any(node_values[key] < self.values[key] for key in node_values.keys())
        )

    def determine_pareto_front(self) -> list:
        """Determines the pareto front, based on childrens metrics.

        Returns:
            list: Pareto front
        """

        # At first all children are in front
        children = list(self.children.values())
        non_dominated_children = []

        # Then we remove them when they are domianted
        for child1 in children:
            # Domination flag
            dominated = False

            for child2 in children:
                if child1 is child2:
                    continue

                if child1.is_dominated(child2):
                    dominated = True
                    break

            if not dominated:
                non_dominated_children.append(child1)

        return non_dominated_children
