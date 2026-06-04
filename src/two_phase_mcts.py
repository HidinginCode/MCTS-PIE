"""Two-phase MCTS orchestrator.

Phase 1: search only over movement (path-only action space) to produce a
Pareto front of near-shortest paths. Phase 2: for each path, run MCTS on the
shift action space only, keeping movement fixed along the phase-1 path.

Works for both single-objective :class:`MctsTree` and the multi-objective
:class:`MOMctsTree` variants — the same orchestrator is used, swapping only
the underlying tree class.
"""

from __future__ import annotations
from typing import Optional
import copy
import random

from controller import Controller
from environment import Environment
from node import Node
from mc_tree import MctsTree
from mo_mc_tree import MOMctsTree
from helper import Helper
from strategies.rollout import RolloutStrategy
from strategies.tree_selection import TreeSelectionStrategy, UCBSelection
from strategies.root_selection import HVRootSelection


# ---------------------------------------------------------------------------
# Controllers that restrict the action space.
# ---------------------------------------------------------------------------


class PathOnlyController(Controller):
    """Controller whose valid actions ignore shifts (shift_dir always (0,0) no-op).

    Movement still happens, and we mark the shift direction as a no-op by
    picking any valid (move_dir, shift_dir) pair where shifting in-place would
    simply leave weights alone — but since the base :meth:`Controller.move`
    always performs the shift, we instead fix shift_dir to a direction that
    lands on the agent's *previous* cell (which is now empty), making the
    shift a no-op for weight accumulation.

    Concretely, phase 1 uses a "dummy" shift: the controller still tracks
    ``_weight_shifted`` but its effect on the map is immaterial for phase 2,
    which will replay the path on a fresh environment.
    """

    def get_all_valid_pairs(self) -> list[tuple]:
        # Expose only one shift option per move direction to shrink branching
        # in phase 1. The shift direction is picked deterministically as the
        # reverse of the move (lands on the cell just vacated).
        dim = self._environment._env_dim
        cx, cy = self._current_pos
        pairs = []
        for dx_m, dy_m in self.DIRECTIONS:
            x_m, y_m = cx + dx_m, cy + dy_m
            if not (0 <= x_m < dim and 0 <= y_m < dim):
                continue
            # Shift back toward old cell (cx, cy): shift_dir = (-dx_m, -dy_m)
            dx_s, dy_s = -dx_m, -dy_m
            x_s, y_s = x_m + dx_s, y_m + dy_s
            if 0 <= x_s < dim and 0 <= y_s < dim:
                pairs.append(((dx_m, dy_m), (dx_s, dy_s)))
            else:
                # Fallback: any in-bounds shift.
                for dx2, dy2 in self.DIRECTIONS:
                    if 0 <= x_m + dx2 < dim and 0 <= y_m + dy2 < dim:
                        pairs.append(((dx_m, dy_m), (dx2, dy2)))
                        break
        return pairs


class FixedPathController(Controller):
    """Controller that replays a pre-computed movement path one step at a time.

    The action space at each step is restricted to ``(fixed_move_dir, shift_*)``
    pairs — the agent cannot deviate from the chosen path. Useful for Phase 2
    of two-phase search and for the A*+MCTS hybrid.
    """

    def __init__(self, environment: Environment, start_pos: tuple,
                 path_moves: list[tuple]) -> None:
        super().__init__(environment=environment, start_pos=start_pos)
        # List of (dx, dy) movement deltas the agent must execute in order.
        self._path_moves = [tuple(m) for m in path_moves]
        self._path_idx = 0

    def clone(self) -> "FixedPathController":
        cloned_env = self._environment.clone()
        clone = FixedPathController(cloned_env, self._start_pos, self._path_moves)
        clone._current_pos = tuple(self._current_pos)
        clone._step_count = int(self._step_count)
        clone._weight_shifted = float(self._weight_shifted)
        clone._distance_to_goal = float(self._distance_to_goal)
        clone._checkpoint_idx = int(self._checkpoint_idx)
        clone._path_idx = int(self._path_idx)
        return clone

    def get_all_valid_pairs(self) -> list[tuple]:
        # Out of moves -> force distance to zero by returning empty actions
        # (tree_policy treats that as terminal via is_terminal_state).
        if self._path_idx >= len(self._path_moves):
            return []
        move_dir = self._path_moves[self._path_idx]
        dim = self._environment._env_dim
        cx, cy = self._current_pos
        x_m, y_m = cx + move_dir[0], cy + move_dir[1]
        if not (0 <= x_m < dim and 0 <= y_m < dim):
            return []
        pairs = []
        for dx_s, dy_s in self.DIRECTIONS:
            if 0 <= x_m + dx_s < dim and 0 <= y_m + dy_s < dim:
                pairs.append((move_dir, (dx_s, dy_s)))
        return pairs

    def move(self, move_dir: tuple, shift_dir: tuple) -> bool:
        ok = super().move(move_dir, shift_dir)
        if ok:
            self._path_idx += 1
        return ok


# ---------------------------------------------------------------------------
# Rollouts for each phase.
# ---------------------------------------------------------------------------


class PathOnlyRollout(RolloutStrategy):
    """Random-walk rollout on the path-only action space (phase 1)."""

    name = "path_only_rollout"

    def rollout(self, tree, leaf, simulations, maximum_moves, remaining_budget):
        used = 0
        results = []
        for _ in range(simulations):
            copy_leaf = leaf.clone()
            ctrl = copy_leaf._controller
            for _ in range(maximum_moves):
                if copy_leaf.is_terminal_state() or used >= remaining_budget:
                    break
                valid = ctrl.get_all_valid_pairs()
                if not valid:
                    break
                mv, sh = random.choice(valid)
                ctrl.move(mv, sh)
                used += 1
            copy_leaf.refresh_values()
            results.append(copy_leaf)
            if used >= remaining_budget:
                break
        pareto = Helper.determine_pareto_front_from_nodes(results)
        leaf._values = dict(random.choice(pareto)._values)
        return used


class WeightOnlyRollout(RolloutStrategy):
    """Random rollout on the shift action space, movement fixed by the path."""

    name = "weight_only_rollout"

    def rollout(self, tree, leaf, simulations, maximum_moves, remaining_budget):
        used = 0
        results = []
        for _ in range(simulations):
            copy_leaf = leaf.clone()
            ctrl = copy_leaf._controller
            for _ in range(maximum_moves):
                if copy_leaf.is_terminal_state() or used >= remaining_budget:
                    break
                valid = ctrl.get_all_valid_pairs()
                if not valid:
                    break
                # Prefer shifts onto low-weight cells to avoid amplifying the
                # total weight accumulated.
                env = ctrl._environment._environment
                mv, sh = min(valid, key=lambda p: env[(ctrl._current_pos[0] + p[0][0] + p[1][0])][(ctrl._current_pos[1] + p[0][1] + p[1][1])])
                ctrl.move(mv, sh)
                used += 1
            copy_leaf.refresh_values()
            results.append(copy_leaf)
            if used >= remaining_budget:
                break
        pareto = Helper.determine_pareto_front_from_nodes(results)
        leaf._values = dict(random.choice(pareto)._values)
        return used


# ---------------------------------------------------------------------------
# Orchestrator.
# ---------------------------------------------------------------------------


class TwoPhaseSearch:
    """Run a path-first, weights-second MCTS/MO-MCTS pipeline.

    Parameters
    ----------
    multi_objective: When True, both phases use :class:`MOMctsTree` and its
        global Pareto archive. Otherwise, plain :class:`MctsTree`.
    """

    def __init__(self, environment: Environment, start_pos: tuple,
                 seed: int = 420, multi_objective: bool = False,
                 max_pareto_path_archive: int = 20) -> None:
        self._environment = environment
        self._start_pos = start_pos
        self._seed = seed
        self._multi_objective = multi_objective
        self._max_pareto = max_pareto_path_archive

    # -- phase 1 ----------------------------------------------------------

    def run_phase1(self, budget: int, per_sim_budget: int,
                   simulations_per_child: int,
                   tree_selection: Optional[TreeSelectionStrategy] = None) -> list[list[tuple]]:
        """Return a list of near-shortest paths (each a list of move deltas).

        Extracted from the global Pareto archive of phase-1 MCTS. Exactly
        one path per Pareto-optimal terminal solution.
        """
        ctrl = PathOnlyController(self._environment, start_pos=self._start_pos)
        root = Node(controller=ctrl)
        tree_cls = MOMctsTree if self._multi_objective else MctsTree
        tree = tree_cls(root=root, seed=self._seed, max_solutions=self._max_pareto)

        tree_sel = tree_selection or UCBSelection()
        tree.search(
            total_budget=budget,
            per_sim_budget=per_sim_budget,
            simulations_per_child=simulations_per_child,
            rollout_func=PathOnlyRollout(),
            root_selection=HVRootSelection(),
            tree_selection=tree_sel,
        )

        # Collect tree-walked paths to terminal nodes. We deliberately do NOT
        # read from MOMctsTree._global_archive here: archive entries pair
        # rollout-terminal *values* with the path to the tree leaf where the
        # rollout *started*, so replaying that path stops short of the goal.
        # The terminal-DFS works for both single- and multi-objective trees.
        raw_paths = self._collect_terminal_paths(tree._root)

        # Reduce to unique move-sequences.
        unique = {}
        for p in raw_paths:
            moves = tuple(step[0][0] for step in p)  # step = ((move, shift), vals)
            if moves and moves not in unique:
                unique[moves] = list(moves)
        return list(unique.values())

    @staticmethod
    def _collect_terminal_paths(root: Node) -> list[list[tuple]]:
        paths = []

        def dfs(node, trail):
            if node.is_terminal_state():
                paths.append(list(trail))
                return
            for key, child in node._children.items():
                dfs(child, trail + [(key, dict(child._values))])

        dfs(root, [])
        return paths

    # -- phase 2 ----------------------------------------------------------

    def run_phase2(self, path_moves: list[tuple],
                   budget: int, per_sim_budget: int,
                   simulations_per_child: int) -> list[dict]:
        """Run weight-pushing MCTS along ``path_moves`` and return Pareto values."""
        ctrl = FixedPathController(self._environment, self._start_pos, path_moves)
        root = Node(controller=ctrl)
        # Always use the MO tree so we can extract the global archive cleanly.
        tree = MOMctsTree(root=root, seed=self._seed, max_solutions=self._max_pareto)

        tree.search(
            total_budget=budget,
            per_sim_budget=per_sim_budget,
            simulations_per_child=simulations_per_child,
            rollout_func=WeightOnlyRollout(),
            root_selection=HVRootSelection(),
            tree_selection=UCBSelection(),
        )

        return [dict(v) for _, v in tree._global_archive]

    # -- full pipeline ----------------------------------------------------

    def run(self, phase1_budget: int, phase2_budget_per_path: int,
            per_sim_budget: int, simulations_per_child: int) -> list[dict]:
        """Execute both phases and return a merged Pareto front of objective dicts."""
        paths = self.run_phase1(phase1_budget, per_sim_budget, simulations_per_child)
        all_values: list[dict] = []
        for path in paths:
            all_values.extend(
                self.run_phase2(path, phase2_budget_per_path, per_sim_budget, simulations_per_child)
            )
        if not all_values:
            return []
        # Final Pareto filter via Helper: build dummy-node wrappers.
        class _W:
            def __init__(self, v):
                self._values = v
                self._ucb_values = v
        wrapped = [_W(v) for v in all_values]
        front = Helper.determine_pareto_front_from_nodes(wrapped)
        return [w._values for w in front]
