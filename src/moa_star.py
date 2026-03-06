"""This module holds the required code for multi objective A*."""

from __future__ import annotations
from controller import Controller
from environment import Environment
import heapq
from collections import defaultdict
import os
import pickle
import math

class MOA_Star_Node:
    """
    Label used by MOA*.
    Represents one Pareto cost label for a specific state.
    """

    def __init__(self, controller: Controller, parent: "MOA_Star_Node | None", move=None):
        self.controller = controller

        # true accumulated cost
        self.g = (self.controller._step_count, self.controller._weight_shifted)

        # heuristic lower bound
        self.h = self.heuristic()

        # evaluation vector
        self.f = tuple(gi + hi for gi, hi in zip(self.g, self.h))

        # backtracking
        self.parent = parent
        self.move = move
    
    def heuristic(self):
        controller = self.controller
        manhattan = controller.calculate_distance_to_goal()

        if not controller._goal_collected:
            weight = controller.weight_to_goal[
                controller.current_pos[0]
            ][controller.current_pos[1]]
        else:
            weight = controller.weight_to_start[
                controller._current_pos[0]
            ][controller._current_pos[1]]

        # slight coupling
        manhattan_adjusted = manhattan + 0.05 * weight

        return (manhattan_adjusted, weight)

    def state_key(self) -> tuple:
        """Returns the state key of the current node.

        Returns:
            tuple: State key
        """
        return (self.controller._current_pos, self.controller._goal_collected) #self.controller.obstacle_signature())


def dominates(a, b):
    return (
        a[0] <= b[0] and
        a[1] <= b[1] and
        (a[0] < b[0] or a[1] < b[1])
    )

class OpenList:
    def __init__(self):
        self._global = []
        self._by_state = defaultdict(list)

    def __len__(self):
        return len(self._global)

    def insert(self, label: MOA_Star_Node) -> bool:
        """Skyline insert method to insert new lables

        Args:
            label (MOA_Star_Node): Node to insert

        Returns:
            bool: Was it inserted or not?
        """
        state = label.state_key()

        # 1) if dominated by existing label → reject
        for other in self._by_state[state]:
            if dominates(other.f, label.f):
                return False

        # 2) find labels dominated by new label
        dominated = [
            other for other in self._by_state[state]
            if dominates(label.f, other.f)
        ]

        # 3) remove dominated labels from BOTH structures
        for other in dominated:
            self._by_state[state].remove(other)
            self._global.remove(other)

        # 4) insert new label
        self._by_state[state].append(label)
        self._global.append(label)
        return True
    
    def pop(self) -> MOA_Star_Node:
        """Removes best node from open.

        Returns:
            MOA_Star_Node: Best node
        """
        best = min(self._global, key=lambda L: L.f)

        self._global.remove(best)

        state = best.state_key()

        # safe removal (avoid crash if already removed elsewhere)
        if best in self._by_state[state]:
            self._by_state[state].remove(best)

        return best
    def cap(self, max_size: int = 20000):
        """Limit total number of labels stored in OPEN."""
        
        if len(self._global) <= max_size:
            return

        # sort global pool by scalarized f
        self._global.sort(key=lambda n: n.f[0] + n.f[1])

        # remove worst nodes
        to_remove = self._global[max_size:]
        self._global = self._global[:max_size]

        # IMPORTANT: also remove them from per-state storage
        for node in to_remove:
            state = node.state_key()
            if state in self._by_state and node in self._by_state[state]:
                self._by_state[state].remove(node)

class ClosedList:
    def __init__(self, max_labels_per_state: int = 5):
        self._data = defaultdict(list)
        self._max_labels = max_labels_per_state
    
    def __len__(self):
        return sum(len(v) for v in self._data.values())

    def is_dominated(self, label: MOA_Star_Node) -> bool:
        """Checks whether a node needs to be pruned"""
        state = label.state_key()

        for g_old in self._data[state]:
            if dominates(g_old, label.g):
                return True

        return False
    
    def insert(self, label: MOA_Star_Node) -> None:
        """Insert label into CLOSED with skyline pruning and cap."""
        state = label.state_key()
        labels = self._data[state]

        labels[:] = [
            g_old for g_old in labels
            if not dominates(label.g, g_old)
        ]

        labels.append(label.g)
        if len(labels) > self._max_labels:
            # keep most promising labels
            # simple and effective score
            labels.sort(key=lambda g: g[0] + g[1])
            del labels[self._max_labels:]

def create_sucessors(parent: MOA_Star_Node) -> list[MOA_Star_Node]:

    valid_pairs = parent.controller.get_all_valid_pairs()
    successors = []

    px, py = parent.controller.current_pos
    env = parent.controller.environment.environment

    for move, shift in valid_pairs:

        dx_m, dy_m = move
        nx, ny = px + dx_m, py + dy_m

        # collapse shifts when no weight
        if env[nx][ny] == 0:
            # skip all shift variations
            shift = (0, 0)
            controller_copy = parent.controller.clone()
            controller_copy.move(move_dir=move, shift_dir=(0,0))
            successors.append(MOA_Star_Node(controller_copy, parent, (move, (0,0))))
            continue

        controller_copy = parent.controller.clone()
        controller_copy.move(move_dir=move, shift_dir=shift)

        successors.append(
            MOA_Star_Node(controller_copy, parent, (move, shift))
        )

    return successors
        
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

def prune_open_with_solutions(open_list, solutions):
    if not solutions:
        return

    open_list[:] = [
        node for node in open_list
        if not any(dominates(sol.g, node.f) for sol in solutions)
    ]

def moa_star(start: tuple, goal: tuple, env_dim: int, heuristic = None, map_type: str = "random_map"):
    """Implementation of multi objective A-Star

    Args:
        start (tuple): Start position as tuple
        get_neighbors (function): Function that gets all neighbors if a node
        heuristic (function): Function that defines the used heuristic
        map_type (str): Name of the maps
    """

    open = OpenList()
    closed = ClosedList()
    solutions = []

    # Create start node -> Needs controller, g, h and parent
    start_environment = Environment(env_dim=env_dim, goal=goal, map_type=map_type, start_pos=start)
    start_controller = Controller(environment=start_environment, start_pos=start)
    start_node = MOA_Star_Node(controller=start_controller, parent=None, move=None)

    #Insert start node into open
    open.insert(start_node)

    while len(open) > 0:
        if solutions and all(
            any(dominates(sol.g, node.f) for sol in solutions)
            for node in open._global
        ):
            break
        print("OPEN:", len(open._global), "CLOSED:", len(closed))
        current = open.pop()

        if closed.is_dominated(current):
            continue
        
        closed.insert(current)

        # Test if current is goal node
        current_controller = current.controller
        if current_controller._goal_collected and current_controller._current_pos == current_controller._start_pos:
            # check if dominated by existing solution -> needs to check g because we are interested in current state values
            dominated = any(dominates(sol.g, current.g) for sol in solutions)
            if dominated: continue

            # Remove solutions that are dominated by current
            solutions = [sol for sol in solutions if not dominates(current.g, sol.g)]

            solutions.append(current)
            prune_open_with_solutions(open, solutions)
            continue

        # Expand the succressors
        sucessors = create_sucessors(parent=current)

        for child in sucessors:
            # Check if child is dominated by any solution
            if any(dominates(sol.g, child.g) for sol in solutions):
                continue
            
            open.insert(child)
        
        open.cap()
            
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
