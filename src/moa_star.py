"""This module holds the required code for multi objective A*."""

from __future__ import annotations
from controller import Controller
from environment import Environment
import heapq
from collections import defaultdict
import os
import pickle

def dominates(cost_1: tuple, cost_2: tuple) -> bool:
    """Determines if cost_1 pareto dominates cost_2.

    Args:
        cost_1 (dict): Cost dict 1
        cost_2 (dict): Cost dict 2

    Returns:
        bool: Domination status
    """
    return all(a <= b for a, b in zip(cost_1, cost_2)) and any(a < b for a, b in zip(cost_1, cost_2))

class MOA_Star_Node:
    """Symbolizes a node in MOA-Star."""

    def __init__(self, controller: Controller, g: tuple, h: tuple, parent: MOA_Star_Node | None):
        """Init method for the MOA-Star node.

        Args:
            controller (Controller): Controller that has all information about state
            g (tuple): Cost vector
            h (tuple): Heurisitc cost vector
            parent (MOA_Star_Node | None): Parent node
        """
        self._controller = controller
        self._g = g # Steps and weight
        self._h = h # Step heuristic and dist heuristic
        self._parent = parent
        self._f = tuple(gi + hi for gi, hi in zip(g, h)) # Steps taken, weight shifted, estimated_distance to goal
        self._move = None
        self._waypoint_collected = False

def heuristic(controller: Controller) -> tuple:
    """Returns the heuristic for a state encapsulated in a controller.

    Args:
        controller (Controller): Controller with state
    """
    dist = controller.calculate_distance_to_goal() # This is only the same since we use manhattan distance on a grid
    return (0, 0)


def reconstruct_path(node: MOA_Star_Node) -> list:
    """Reconstructs the path from a given end node.

    Args:
        node (MOA_Star_Node): End node for moa-star
    
    Returns:
        list: Reconstructed path from node
    """
    current = node
    path = []

    while current is not None:
        path.append((current._controller._current_pos, current._move))
        current = current._parent
    
    path.reverse()
    return path

def moa_star(start: tuple, goal: tuple, env_dim: int, heuristic = heuristic, map_type: str = "random_map"):
    """Implementation of multi objective A-Star

    Args:
        start (tuple): Start position as tuple
        get_neighbors (function): Function that gets all neighbors if a node
        heuristic (function): Function that defines the used heuristic
        map_type (str): Name of the maps
    """
    print("Starting MOA-Star ...")
    logger = AStarLogger()
    start_evironment = Environment(env_dim, goal, map_type=map_type, start_pos=start)
    start_controller = Controller(start_evironment, start)


    start_node = MOA_Star_Node(start_controller, g=(0,0), h=heuristic(start_controller), parent=None)
    front = [start_node]
    cost_db = defaultdict(list)
    open_db = defaultdict(list)
    solutions = []
    start_state = get_state(start_node)
    open_db[start_state].append(start_node)

    #print("Entering loop ...")
    while front:
        # termination check FIRST
        if solutions and all(
            any(dominates(sol._g, node._f) for sol in solutions)
            for node in front
        ):
            break

        current = current = front.pop(0)
        current: MOA_Star_Node
        
        # Check if goal reached
        dist_to_goal = current._controller.calculate_distance_to_goal()
        current._waypoint_collected = current._controller._goal_collected
        #print(f"At pos {current._controller._current_pos}, dist = {dist_to_goal}, g = {current._g}")

        if dist_to_goal == 0:
            if any(dominates(sol._g, current._g) for sol in solutions):
                continue
            solutions = [s for s in solutions if not dominates(current._g, s._g)]
            solutions.append(current)
            continue

        #print(len(front))
        #input()
        state = (
            current._controller._current_pos,
            current._controller._goal_collected
        )

        if any(dominates(sol._g, current._g) for sol in solutions):
            continue
        
        labels = cost_db[state]

        # Skip if current label is dominated by an existing one
        # Skip if identical cost already exists
        if current._g in labels:
            continue

        # Skip if dominated
        if any(dominates(old, current._g) for old in labels):
            continue

        # Remove existing labels that are dominated by current
        labels[:] = [old for old in labels if not dominates(current._g, old)]

        # Insert current label
        labels.append(current._g)

        # Expand node
        all_valid_moves = current._controller.get_all_valid_pairs()

        for move_dir, shift_dir in all_valid_moves:
            copy_controller = current._controller.clone()
            copy_controller.move(move_dir, shift_dir)
            g_new = (copy_controller.step_count, copy_controller._weight_shifted)
            h_new = heuristic(copy_controller)
            child = MOA_Star_Node(copy_controller, g_new, h_new, current)
            child._move = (move_dir, shift_dir)
            insert_open(front, open_db, child)
            #print("Added child")
    
    solutions = pareto_filter(solutions)
    for i, solution in enumerate(solutions):
        logger.log(solution, i)
    paths = [reconstruct_path(sol) for sol in solutions]
    return paths

def pareto_filter(nodes: list[MOA_Star_Node]) -> list:
    """Pareto filter for MOA-Star nodes.

    Args:
        nodes (list(MOA_Star_Node)): Nodes to filter

    Returns:
        list: Pareto front
    """
    #print(f"Worker {os.getpid()} entered the pareto filter ...")
    pareto = []
    for node in nodes:
        dominated = False
        for other in nodes:
            if other is not node and dominates(other._g, node._g):
                dominated = True
                break
        if not dominated:
            pareto.append(node)
    print(f"Worker {os.getpid()} left the pareto filter ... Hurray")
    return pareto

class AStarLogger():
    """This class contains the logger capabilities for the MOA-Star class."""
    def __init__(self):
        os.makedirs("./moastar_log", exist_ok=True)
    
    @staticmethod
    def log(node: MOA_Star_Node, solution_index: int) -> None:
        """Writes log file for finished nodes.

        Args:
            node (MOA_Star_Node): Goal node
            solution_index (int): Index of the found solution
        """
        os.makedirs(f"./moastar_log/{node._controller._environment._map_type}-{node._controller._environment._env_dim}", exist_ok=True)
        data = {
            "map_name": node._controller._environment._map_type,
            "env_dim": node._controller._environment._env_dim,
            "start": node._controller._start_pos,
            "goal": node._controller._environment._goal,
            "values": node._g,
            "path": reconstruct_path(node),
        }

        with open(f"./moastar_log/{node._controller._environment._map_type}-{node._controller._environment._env_dim}/solution-{solution_index}.pickle", "wb") as f:
            pickle.dump(data, f)

def pop_nondominated(front: list[MOA_Star_Node],
                     open_db: dict) -> MOA_Star_Node:

    for i, node in enumerate(front):
        dominated = False
        for other in front:
            if other is not node and dominates(other._f, node._f):
                dominated = True
                break

        if not dominated:
            # Remove from OPEN DB skyline
            state = get_state(node)
            open_db[state].remove(node)
            return front.pop(i)

    raise RuntimeError("No nondominated node found")

def insert_open(front, open_db, child):

    state = get_state(child)
    labels = open_db[state]

    # If dominated by existing OPEN node of same state → discard
    for node in labels:
        if dominates(node._f, child._f):
            return

    # Find nodes dominated by child
    dominated_nodes = [node for node in labels
                       if dominates(child._f, node._f)]

    # Remove dominated nodes from BOTH structures
    for node in dominated_nodes:
        labels.remove(node)
        front.remove(node)

    # Insert node into skyline
    labels.append(child)

    # Insert into global OPEN
    front.append(child)

def get_state(node: MOA_Star_Node):
    return (
        node._controller._current_pos,
        node._controller._goal_collected
    )