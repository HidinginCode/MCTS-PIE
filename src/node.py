"""This module contains the node class which later builds the MCTS tree."""
from __future__ import annotations
from controller import Controller
import random

class Node():
    """This node class is the basis for later building the MCTS tree."""

    def __init__(self, controller: Controller, parent: Node | None = None, last_move: tuple = None):
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
        self._paths_changed = False # This is for recomputation of HV to make an if statement (if paths_changed update else dont)
        self._old_hv_values = [] # Here old HV values are stored when a recomputation is not necessary. Structure is: ([contrib hv 1, contrib hv 2, ...], full HV)
        self._old_epsilon_candidate = None # Here the last candidate for epsilon domination is stored
        self._old_cd_values = [] # Here old CD values are stored so that recomputation is not necessary
        self._last_move = last_move

    def clone(self) -> Node:
        """Clone method for node object.

        Returns:
            Node: Cloned node
        """
        # No need to be copied
        new_node = Node(self._controller, None)
        new_node._depth = self._depth
        new_node._visits = self._visits
        new_node._last_move = self._last_move
        new_node._paths_changed = bool(self._paths_changed)
        new_node._old_epsilon_candidate = self._old_epsilon_candidate
        new_node._old_cd_values = self._old_cd_values

        # Copy
        new_node._values = self._values.copy()
        new_node._ucb_values = self._ucb_values.copy()


        new_node._children = {}
        new_node._pareto_paths = [
            [entry[0], entry[1].copy()]
            for entry in self._pareto_paths
        ]

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

        # Oszillierende positionen verhindern dadurch dass wir nicht auf die Letzte pos zurück dürfen
        current_pos = self._controller._current_pos
        if self._parent is not None and current_pos != self._controller._environment._goal:
            last_pos = self._parent._controller._current_pos
            bad_move = (last_pos[0] - current_pos[0], last_pos[1]-current_pos[1])
        else:
            bad_move = None
        
        untried_actions = self.get_untried_actions()

        if bad_move is not None:
            untried_actions = [untried_action for untried_action in untried_actions if untried_action[0] != bad_move]

        if untried_actions:
            move_pair = random.choice(untried_actions)
            move_dir, shift_dir = move_pair
            child_node = Node(self._controller, self, move_pair)
            child_node._controller.move(move_dir, shift_dir)
            # After move we load the new objective values into value dict
            child_node.refresh_values()
            self._children[move_pair] = child_node
            return child_node
        else:
            return None