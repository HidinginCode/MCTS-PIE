"""This module represents a the node class for the MCTS tree."""

from __future__ import annotations
import math
import random
from multiprocessing import Pool
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
        return f"Visits: {self.visits}\nValues: {self.values}"

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

    def get_front(self) -> list:
        """Returns the pareto front of the current node.

        Returns:
            list: Pareto front
        """

        return self.front

    def set_front(self, front: list) -> None:
        """Sets a specified front for the node.

        Args:
            front (list): Front to be set
        """

        self.front = front

    def get_all_valid_directions(self) -> list:
        """Returns all valid pairs of movement and shifting 
        direction that would be valid in this state.

        Returns:
            list: Valid pairs
        """
        all_valid_pairs = []
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
                    all_valid_pairs.append((movement_direction, shifting_direction))

        return all_valid_pairs


    def get_untried_directions(self) -> list:
        """Returns all actions that werent tried on this node by now.

        Returns:
            list: Untried actions
        """
        tried_directions = set(self.children.keys())
        return[a for a in self.get_all_valid_directions() if a not in tried_directions]


    def expand(self) -> Node | None:
        """Expands the node, creating new children for an unexpanded direction if there is one.
        (movement_direction, shifting_direction)."""

        if self.state.get_terminal_state():
            return None

        untried_directions = self.get_untried_directions()
        if not untried_directions:
            return None

        random.shuffle(untried_directions)
        for movement_direction, shifting_direction in untried_directions:
            controller = deepcopy(self.state.get_state_controller())
            controller.move_agent(movement_direction.value, shifting_direction.value)
            child = Node(state=State(controller), parent=self)
            self.children[(movement_direction, shifting_direction)] = child

            return child


    def update_node(self, metrics: dict):
        """Update means for each objective.

        Args:
            metrics (dict): Metrics used for the update
        """
        self.visits += 1
        for key in self.values.keys():
            # Calculate the mean for each objective
            float_metric = float(metrics[key])
            self.values[key] += (float_metric - self.values[key]) / self.visits

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

        return all(node_a_metrics[key] >= node_b_metrics[key] for key in node_a_metrics) \
        and any(node_a_metrics[key] > node_b_metrics[key] for key in node_a_metrics)

    def pareto_reduction(self, vectors: list[dict]) -> list[dict]:
        """Removes dominated solutions and creates the pareto front.

        Args:
            vectors (list[dict]): All solution vectors

        Returns:
            list[dict]: Pareto front
        """

        front = []
        for vector in vectors:
            if not any(self.dominates(member, vector) for member in front):
                # Remove members from pareto front dominated by vector
                front = [member for member in front if not self.dominates(vector, member)]
                front.append(vector)
        return front

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

        # Handle unvisited children
        unvisited = [child for child in candidates if child.get_visits() == 0]

        if unvisited:
            return random.choice(unvisited)

        log_parent = math.log(self.get_visits() + 1)

        # Compute min/max metrics among children for normalization
        metrics = list(self.values.keys())
        stats = {
            metric: (min(child.values[metric] for child in candidates),
                    max(child.values[metric] for child in candidates))
            for metric in metrics
        }

        # Compute per-objective UCB
        for child in candidates:

            child.set_ucb_vector({
                metric: (
                    # We flip the scale so smaller = higher score since we want to minimize
                    (stats[metric][1] - child.values[metric]) /
                    (stats[metric][1] - stats[metric][0] + 1e-9)
                    + c * math.sqrt(log_parent / child.visits)
                )
                for metric in metrics
            })

        # Pareto front extraction
        pareto = [
            a for a in candidates
            if not any(self.dominates(b.ucb_vector, a.ucb_vector) for b in candidates)]

        return random.choice(pareto or candidates)

    @staticmethod
    def _simulate_leaf_worker(args: tuple):
        """Wrapper for multiprocessing

        Args:
            args (tuple): Arguments that should be passed into actual method
        """
        self_reference, leaf_node, maximum_depth = args
        return self_reference.multiprocessing_leaf_simulation(leaf_node, maximum_depth)

    def simulate_leaf(
            self,
            leaf_node: Node,
            maximum_depth: int = 15,
            number_of_simulations: int = 10
    ) -> None:
        """Simulates from a given leaf node by applying random valid actions.

        Args:
            leaf_node (Node): Leaf node from which to start the simulation
            maximum_depth (int, optional): Maximum moves that will be simulated. Defaults to 15.
        """

        # Make an independent copy of the node to simulate on
        solutions = []

        with Pool(number_of_simulations) as pool:
            args = [(self, leaf_node, maximum_depth) for _ in range(number_of_simulations)]
            results = pool.map(Node._simulate_leaf_worker, args)

        # Flatten results (list of lists of dicts)
        solutions = [state for rollout in results for state in rollout]

        # Handle case where no moves were possible
        if not solutions:
            solutions.append(leaf_node.get_state().get_state_metrics())

        # Compute average metrics over the simulated rollout
        averages = {
            key: sum(d[key] for d in solutions) / len(solutions)
            for key in solutions[0]
        }

        # Update node values and store Pareto front
        leaf_node.set_value(averages)
        leaf_node.set_front(leaf_node.pareto_reduction(solutions))


    def multiprocessing_leaf_simulation(self, leaf_node: Node, maximum_depth: int) -> list:
        """Multiprocessing function to simulate all actions for a leaf and collect the solutions.

        Args:
            leaf_node (Node): Leaf node from which to simulate
            maximum_depth (int): Maximum simulation depth

        Returns:
            list: solutions
        """
        depth = 0
        solutions = []
        leaf_copy = deepcopy(leaf_node)
        controller = leaf_copy.get_state().get_state_controller()

        while not leaf_copy.get_state().get_terminal_state() and depth < maximum_depth:
            valid_pairs = leaf_copy.get_all_valid_directions()
            if not valid_pairs:
                break

            movement_direction, shifting_direction = random.choice(valid_pairs)
            controller.move_agent(
                move_direction=movement_direction.value,
                shifting_direction=shifting_direction.value
            )

            depth += 1
            current_state = State(state_controller=controller)
            solutions.append(current_state.get_state_metrics())

        return solutions
