"""This module holds the required code for multi objective A*."""

from __future__ import annotations
from controller import Controller
from environment import Environment
import heapq
from collections import defaultdict

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
        self._f = (g[0]+h[0], g[1]) # Steps taken, weight shifted, estimated_distance to goal
        self._move = None
    
    def __lt__(self, other):
        """Heapq tie-breaker. Uses lexicographical ordering"""
        return self._f < other._f

def heuristic(controller: Controller) -> tuple:
    """Returns the heuristic for a state encapsulated in a controller.

    Args:
        controller (Controller): Controller with state
    """
    dist = controller.calculate_distance_to_goal() # This is only the same since we use manhattan distance on a grid
    return (dist, 0)


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
    
    return path

def moa_star(start: tuple, goal: tuple, env_dim: int, heuristic):
    """Implementation of multi objective A-Star

    Args:
        start (tuple): Start position as tuple
        get_neighbors (function): Function that gets all neighbors if a node
        heuristic (function): Function that defines the used heuristic
    """
    print("Starting MOA-Star ...")
    start_evironment = Environment(env_dim, goal)
    start_controller = Controller(start_evironment, start)


    start_node = MOA_Star_Node(start_controller, g=(0,0), h=heuristic(start_controller), parent=None)
    front = [start_node]
    cost_db = defaultdict(list)
    solutions = []

    print("Entering loop ...")
    while front:
        current = heapq.heappop(front)
        current: MOA_Star_Node
        
        # Check if goal reached
        dist_to_goal = current._controller.calculate_distance_to_goal()
        print(f"At pos {current._controller._current_pos}, dist = {dist_to_goal}, g = {current._g}")

        if dist_to_goal == 0:
            print("Goal was reached!")
            solutions.append(current)
            print(f"Found {len(solutions)} solutions. Stopping.")
            continue

        dominated = False
        for f_old in cost_db[current._controller._current_pos]:
            if dominates(f_old, current._f):
                print(f"Node at {current._controller._current_pos} dominated by {f_old}")
                dominated = True
                break

        if dominated:
            continue

        cost_db[current._controller._current_pos].append(current._g)
    
        # Expand node
        all_valid_moves = current._controller.get_all_valid_pairs()

        for move_dir, shift_dir in all_valid_moves:
            copy_controller = current._controller.clone()
            copy_controller.move(move_dir, shift_dir)
            g_new = (copy_controller.step_count, copy_controller._weight_shifted)
            h_new = heuristic(copy_controller)
            child = MOA_Star_Node(copy_controller, g_new, h_new, current)
            child._move = (move_dir, shift_dir)
            heapq.heappush(front, child)
            print("Added child")
    
    paths = [reconstruct_path(sol) for sol in solutions]
    return paths
