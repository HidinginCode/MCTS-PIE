"""This module contains the class that creates and simulates the MCTS tree."""

import math
import random
import multiprocessing as mp
import os
import numpy as np

from tqdm import tqdm
from node import Node
from controller import Controller

class McTree():
    """This class represents the MCTS tree."""

    SHARED_NODE = None

    def __init__(self, root: Node):
        """Init method for McTree.

        Args:
            root (Node): Root node of tree
            max_depth (int): Max depth of tree
        """
        self.root = root
        self.max_depth = None
        self.identificator = id(self)
        mp.set_start_method("fork", force=True)

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

    def select_node(self, root: Node) -> Node | None:
        """Slection method to get a leaf node or one that is not fully expanded.

        Args:
            root (Node): Root node

        Returns:
            Node | None: Leaf or not fully unexpanded node
        """

        # Check if node is fully expanded
        if not root.is_fully_expanded():
            return root

        # Check if root is terminal state (no expansion needed)
        # Return none so we dont have to check for terminal state in later steps
        if root.get_state().get_terminal_state():
            return None

        selected_node = self.select_node(self.ucb_child_selection(root))
        return selected_node

    def child_selection(self, node: Node) -> Node:
        """Selects a child based on the pareto front of children.

        Args:
            node (Node): Node from where child is selected.

        Returns:
            Node: Child node
        """

        # Check if front is available
        if len(node.get_front()) == 0:
            node.set_front(node.determine_pareto_front())

        if len(node.get_front()) == 0:
            print(f"Number of children: {len(node.get_children())}")
            input()

        front = node.get_front()
        visits = [child.get_visits() for child in front]


        if any(v == 0 for v in visits):
            probabilities = [1 / len(front)] * len(front)
        else:
            total_visits = sum(visits)
            probabilities = [v / total_visits for v in visits]

        return random.choices(front, probabilities, k=1)[0]
        #return random.choice(front)

    @staticmethod
    def normalize_unit_vector(values: dict[str, float]) -> dict[str, float]:
        """Normalize a dict of numeric values to a unit vector (L2 norm = 1)."""
        norm = math.sqrt(sum(v ** 2 for v in values.values()))
        if norm == 0:
            return {k: 0.0 for k in values}  # avoid division by zero
        return {k: v / norm for k, v in values.items()}

    def ucb_child_selection(self, node: Node) -> Node:
        """Selects children based on pareto dominance of UCB1-Calculations

        Args:
            node (Node): Node of which children are selected

        Returns:
            Node: Child node
        TODO: Figure out other selection than random choice (Hypervolume, Crowding-Distance,...)
        """
        children = node.get_children().values()
        number_of_children = len(children)

        for child in children:
            child: Node
            child_values = child.get_values()
            dimensions = len(child_values) # Number of dimensions
            child_visits = child.get_visits()
            parent_visits = node.get_visits()
            normalized_values = McTree.normalize_unit_vector(child.get_values())
            exploration_term = np.sqrt(
                (2*np.log(
                    parent_visits*np.sqrt(np.sqrt(dimensions*number_of_children)))
                )/child_visits
            )#from multiprocessing import Pool
            ucb_vec = {k: v + exploration_term for k, v in normalized_values.items()}
            child.set_ucb_vector(ucb_vec)

        pareto_front = Node.determine_pareto_from_ucb(children)
        return random.choice(pareto_front)

    def expand(self, node: Node) -> Node:
        """Creates a child for a given node.

        Args:
            node (Node): Node to expand

        Returns:
            Node: New child node
        """

        # Get untried actions (all are valid)
        untried_actions = node.get_untried_actions()
        move_direction, shift_direction = random.choice(untried_actions)

        # Create new child and move it
        # No need to deepcopy state, it is deepcopied in childs constructor
        child = Node(node.get_state(), node)
        child.get_state().get_state_controller().move_agent(move_direction, shift_direction)
        child.values = child.get_state().get_state_metrics()
        child.parent_actions = (move_direction.name, shift_direction.name)
        node.children[(move_direction, shift_direction)] = child

        return child

    def simulate_leaf(self, leaf: Node, number_of_simulations: int, maximum_moves: int) -> None:
        """Function that simulates random rollouts from a leaf.

        Args:
            leaf (Node): Leaf node
            number_of_simulations (int): Number of simulations
            maximum_moves (int): Maximum number of moves per simulation
        """

        McTree.SHARED_NODE = leaf

        with mp.Pool(
            processes = min(os.cpu_count(), number_of_simulations),
        ) as pool:
            it = pool.imap_unordered(
                McTree.multiprocess_heavy_distance_rollout,
                [maximum_moves] * number_of_simulations,
            )
            results = list(it)

        if results:
            non_dominated_results = leaf.determine_pareto_from_list(results)
            leaf.set_values(random.choice(non_dominated_results).get_values())

    @staticmethod
    def iterative_leaf_simulation(leaf: Node, number_of_sims: int, maximum_depth: int)->list[Node]:
        """Method that iteratively does the leaf simulations.

        Args:
            leaf (Node): Leaf for start of simulations
            number_of_sims (int): Number of simulations
            maximum_depth (int): Maximum number of views

        Returns:
            Node: Simulation solutions
        """

        solutions = []

        for _ in range(number_of_sims):
            leaf_copy = leaf.clone()
            copy_controller = leaf_copy.get_state().get_state_controller()
            for _ in range(maximum_depth):
                if leaf_copy.get_state().get_terminal_state():
                    break
                valid_moves = leaf_copy.get_all_valid_actions()
                move_direction, shift_direction= random.choice(valid_moves)
                copy_controller.move_agent(move_direction, shift_direction)

            solutions.append(leaf_copy)

        return solutions

    @staticmethod
    def multiprocess_light_rollout(maximum_moves: int) -> Node:
        """Multiprocessing light rollout for leaf simulation.

        Args:
            maxmimum_moves(int): Maximum number of moves untill break.

        Returns:
            Node: Copy of node after simulations
        """
        leaf_copy = Node(McTree.SHARED_NODE.state.clone(), None)
        copy_controller = leaf_copy.state.state_controller

        for _ in range (maximum_moves):

            # Check if terminal state was reached
            if leaf_copy.get_state().get_terminal_state():
                break

            valid_moves = leaf_copy.get_all_valid_actions()
            move_direction, shift_direction= random.choice(valid_moves)
            copy_controller.move_agent(move_direction, shift_direction)

        return leaf_copy

    @staticmethod
    def multiprocess_heavy_distance_rollout(maximum_moves: int) -> Node:
        """Heavy rollout for leaf simulation.
        Moves are selected by either decreasing or staying at the same distance to the goal.

        Args:
            maximum_moves (int): Number of maximum moves till break

        Returns:
            Node: Simulated node
        """

        leaf_copy = Node(McTree.SHARED_NODE.state.clone(), None)
        copy_controller = leaf_copy.state.state_controller
        goal = copy_controller.map_copy.goal
        manhattan = lambda p, q: abs(p[0]-q[0]) + abs(p[1]-q[1])

        for _ in range(maximum_moves):
            # Break if we reached terminal state
            if leaf_copy.get_state().get_terminal_state():
                break

            agent_pos = copy_controller.current_agent_position
            # Get needed parts of calculation and prepare move list
            current_distance_to_goal = copy_controller.calculate_distance_to_goal()
            distance_minimizing_moves = []
            valid_moves = leaf_copy.get_all_valid_actions()

            # Get moves that do not increase distance
            for move_dir, shifting_dir in valid_moves:
                new_pos = (agent_pos[0] + move_dir.value[0],
                           agent_pos[1] + move_dir.value[1])
                new_distance_to_goal = Controller.remaining_roundtrip_distance(new_pos, copy_controller.start_pos, copy_controller.map_copy.goal, copy_controller.goal_collected)
                if new_distance_to_goal <= current_distance_to_goal:
                    distance_minimizing_moves.append((move_dir, shifting_dir))

            # Randomly chose from moves
            move_direction, shift_direction = random.choice(distance_minimizing_moves)
            copy_controller.move_agent(move_direction, shift_direction)
        return leaf_copy

    @staticmethod
    def multiprocess_heavy_pareto_rollout(maximum_moves: int) -> Node:
        """Heavy rollout that uses pareto dominance to select the next move.

        Args:
            maximum_moves (int): Maximum number of tested moves

        Returns:
            Node: Node to simulate
        """

        def dominates(tuple1: tuple, tuple2: tuple) -> bool:
            """Determines if move 1 is dominates the other

            Args:
                tuple1 (tuple): Move to be checked if dominates
                tuple2 (tuple): Move to be checked if dominated

            Returns:
                bool: Domination status
            """
            return(
                (tuple1[0] <= tuple2[0] and tuple1[1] <= tuple2[1] and tuple1[2] <= tuple2[2]) and
                (tuple1[0] < tuple2[0] or tuple1[1] < tuple2[1] or tuple1[2] < tuple2[2])
            )

        
        leaf_copy = Node(McTree.SHARED_NODE.state.clone(), None)
        copy_controller = leaf_copy.state.state_controller
        manhattan = lambda p, q: abs(p[0]-q[0]) + abs(p[1]-q[1]) # Fancy lambda expression to calculate manhattan distance
        map_list = copy_controller.map_copy.map
        agent = copy_controller.current_agent
        goal = copy_controller.map_copy.goal

        for _ in range(maximum_moves):
            if leaf_copy.get_state().get_terminal_state():
                break

            valid_moves = leaf_copy.get_all_valid_actions()
            current_pos = copy_controller.current_agent_position
            front: list[tuple] = []

            for move_dir, shift_dir in valid_moves:
                new_pos = (current_pos[0] + move_dir.value[0],
                        current_pos[1] + move_dir.value[1])

                obstacle_weight = map_list[new_pos[0]][new_pos[1]]
                candidate = (
                    agent.step_count + 1,
                    manhattan(new_pos, goal),
                    agent.weight_shifted + obstacle_weight,
                    (move_dir, shift_dir),
                )

                # Check if candidate is dominated by any member of the front
                dominated = False
                for f in front:
                    if dominates(f, candidate):
                        dominated = True
                        break
                if dominated:
                    continue

                # Remove members of the front that are dominated by the candidate
                i = 0
                while i < len(front):
                    if dominates(candidate, front[i]):
                        front.pop(i)
                    else:
                        i += 1
                front.append(candidate)

            move_dir, shift_dir = random.choice(front)[3]
            copy_controller.move_agent(move_dir, shift_dir)
        return leaf_copy

    @staticmethod
    def multiprocess_heavy_minweight_rollout(maximum_moves: int) -> Node:
        """Heavy rollout method that picks the lowest weight from the neighborhood to move to and shift.
        The only viable moves for this are those that minimize distance to goal.

        Args:
            maximum_moves (int): Maximum number of moves before break

        Returns:
            Node: Simulated node
        """
        leaf_copy = Node(McTree.SHARED_NODE.state.clone(), None)
        copy_controller = leaf_copy.state.state_controller
        manhattan = lambda p, q: abs(p[0]-q[0]) + abs(p[1]-q[1])
        map_list = copy_controller.map_copy.map
        goal = copy_controller.map_copy.goal

        for _ in range(maximum_moves):
            # Break if we reached terminal state
            if leaf_copy.get_state().get_terminal_state():
                break

            # Get needed parts of calculation and prepare move list
            current_position = copy_controller.current_agent_position
            current_distance_to_goal = manhattan(current_position, goal)
            valid_moves = leaf_copy.get_all_valid_actions()
            smallest_weight = np.inf # Large number to be set later
            selected_move = None

            # Get moves that do not increase distance
            for move_dir, shift_dir in valid_moves:
                new_pos = (current_position[0] + move_dir.value[0],
                            current_position[1] + move_dir.value[1])
                #print(f"New Pos: {new_pos}, Current Pos: {current_position}")
                new_distance_to_goal = manhattan(new_pos, goal)
                if new_distance_to_goal <= current_distance_to_goal:
                    if map_list[new_pos[0]][new_pos[1]] < smallest_weight:
                        selected_move = (move_dir, shift_dir)

            copy_controller.move_agent(selected_move[0], selected_move[1])
        return leaf_copy

    def backpropagate(self, node: Node) -> None:
        """Backpropagates metrics from the new child node through the parents.
        This makes the parents metrics an average of the metrics of their children.

        Args:
            node (Node): Root of backpropagation
        """

        current_node = node
        while current_node is not None:
            current_node.increase_visits(1)
            # Check if current node has any children
            if current_node.get_children():

                # Iterate over each key/value
                for key in current_node.get_values().keys():
                    value = 0
                    for child in current_node.get_children().values():
                        value += child.get_values()[key]
                    value/=len(current_node.get_children())
                    current_node.set_value(key, value)

            current_node = current_node.get_parent()

    def run_search(self, iterations: int = 1) -> None:
        """Runs the MCTS search for a specified number of iterations.

        Args:
            iterations (int, optional): Number of times to run MCTS. Defaults to 1.
        """

        solutions = []

        for _ in tqdm(range(iterations)):

            leaf = self.select_node(self.root)

            if leaf is not None:

                if leaf.state.get_terminal_state(): 
                    if leaf not in solutions:
                        solutions.append(leaf)
                    continue

                new_child = self.expand(leaf)
                self.simulate_leaf(new_child, 16, 200)
                self.backpropagate(new_child)

        solutions = Node.determine_pareto_from_list(solutions)
        for solution in solutions:
            node = solution
            path = []
            actions = []
            while node is not None:
                path.append(node.get_state().get_state_controller().get_current_agent_position())
                actions.append(node.get_parent_actions())
                node = node.get_parent()
            print("####################################################################")
            path.reverse()
            actions.reverse()
            print(path)
            print(actions)
