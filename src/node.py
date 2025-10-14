"""This module represents a the node class for the MCTS tree."""

from __future__ import annotations
import math
import random
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
        self.parent = parent
        self.children = {}
        self.identificator = id(self)
        self.visits = 0
        self.values = {
            "energy_consumed": 0.0,
            "step_count": 0,
            "weight_shifted": 0.0,
            "amount_of_shifts": 0
        }
        self.ucb_vector = {}

    def __str__(self):
        """String method for the node class."""
        return f"Visits: {self.visits}\nValues: {self.values}"

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

    def set_value(self, values: dict) -> None:
        """Sets the value to a specified amount.

        Args:
            value (float): New value
        """

        self.values = values

    def get_ucb_vector(self) -> dict:
        """Returns the ucb vector of the node.

        Returns:
            dict: Ucb vector
        """

        return self.ucb_vector

    def set_ucb_vector(self, new_vector: dict) -> None:
        """Sets a new ucb vector for a node.

        Args:
            new_vector (dict): New ucb vector
        """

        self.ucb_vector = new_vector

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
                    move_direction=movement_direction.value,
                    shifting_direction=shifting_direction.value
                )

                if move_valid:
                    new_state = State(state_controller = copy_controller)
                    new_node = Node(state=new_state, parent=self)

                    self.children[(movement_direction, shifting_direction)] = new_node
                else:
                    self.children[(movement_direction, shifting_direction)] = None

    def update_node(self, metrics: dict):
        """Update means for each objective.

        Args:
            metrics (dict): Metrics used for the update
        """
        self.visits += 1
        for key in self.values.keys():
            # Calculate the mean for each objective
            float_metric = float(metrics[key])
            self.values[key]+= (float_metric - self.values[key]) / self.visits

    def dominates(
            self,
            node_a_metrics: dict[str, float|int],
            node_b_metrics: dict[str, float|int]
        ) -> bool:
        """Tests for dominance and returns true if vector a dominates vector b

        Args:
            node_a_metrics (dict[str, float | int]): Metrics of the first node
            node_b_metrics (dict[str, float | int]): Metrics of the second node

        Returns:
            bool: Domination truth value
        """

        return all(node_a_metrics[key] <= node_b_metrics[key] for key in node_a_metrics) \
        and any(node_a_metrics[key] < node_b_metrics[key] for key in node_a_metrics)

    def select_child_pareto_ucb(self, c: float = 1.4) -> Node:
        """Method for selecting a child from the ucb pareto front.

        Args:
            c (float, optional): C value. Defaults to 1.4.

        Returns:
            Node: Selected child
        """
        assert self.children, (
            "No children to select from."
        )
        # Exclude invalid placeholder children
        candidates = [child for child in self.children.values() if child is not None]

        log_parent = math.log(max(1, self.visits))

        # Compute min/max metrics among children for normalization
        metrics = list(self.values.keys())
        stats = {
            metric: (min(child.values[metric] for child in candidates),
                    max(child.values[metric] for child in candidates))
            for metric in metrics
        }

        # Compute per-objective UCB
        for child in candidates:

            # Safety check for children with no visits
            if child.visits == 0:
                child.set_ucb_vector({m: -float("inf") for m in metrics})
                continue

            child.set_ucb_vector({
                metric: (
                    (0.0 if stats[metric][1] == stats[metric][0]
                    else (child.values[metric] - stats[metric][0]) /
                    (stats[metric][1] - stats[metric][0]))
                    - c * math.sqrt(log_parent / child.visits)
                )
            for metric in metrics
            })

        # Pareto front extraction
        pareto = [
            a for a in candidates
            if not any(self.dominates(b.ucb_vector, a.ucb_vector) for b in candidates)]

        return random.choice(pareto or candidates)
