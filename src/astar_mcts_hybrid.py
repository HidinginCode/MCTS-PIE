"""A* + MCTS hybrid.

Phase 1 uses :class:`A_Star` to compute the Pareto front of near-shortest
movement paths via epsilon-constraint search. Phase 2 replays each path
through :class:`FixedPathController` and runs MCTS purely on the shift action
space to optimize weight pushing.

This is a thin orchestrator built on top of the Phase D infrastructure in
:mod:`two_phase_mcts`.
"""

from __future__ import annotations

from astar import A_Star
from environment import Environment
from helper import Helper
from two_phase_mcts import TwoPhaseSearch


class AStarMctsHybrid:
    """Combine A*-computed paths with MCTS-optimized weight pushing."""

    def __init__(self, map_name: str, env_dim: int, start: tuple, goal: tuple,
                 checkpoints: list = None, seed: int = 420,
                 multi_objective: bool = False,
                 max_pareto_path_archive: int = 20) -> None:
        self._astar = A_Star(map_name, env_dim, start, goal, checkpoints=checkpoints)
        self._env = self._astar.env
        self._start = start
        self._seed = seed
        self._multi_objective = multi_objective
        self._max_pareto = max_pareto_path_archive

    # ------------------------------------------------------------------

    def _extract_moves(self, positions: list[tuple]) -> list[tuple]:
        moves = []
        for i in range(len(positions) - 1):
            dx = positions[i + 1][0] - positions[i][0]
            dy = positions[i + 1][1] - positions[i][1]
            moves.append((dx, dy))
        return moves

    def run(self, phase2_budget_per_path: int, per_sim_budget: int,
            simulations_per_child: int) -> list[dict]:
        """Run the hybrid and return a merged Pareto front of objective dicts."""
        # Phase 1: A* Pareto front of (steps, weight-lower-bound, path positions).
        astar_l_star = self._astar.compute_shortest_steps()
        max_epsilon = self._astar.env_dim * 3
        paths_positions: list[list[tuple]] = []
        seen = set()
        for epsilon in range(max_epsilon + 1):
            step_limit = astar_l_star + epsilon
            node = self._astar.minimize_weight_with_step_bound(step_limit)
            if node is None:
                continue
            positions = tuple(self._astar.reconstruct_path(node))
            if positions in seen:
                continue
            seen.add(positions)
            paths_positions.append(list(positions))

        # Phase 2: MCTS weight-pushing for each path.
        orchestrator = TwoPhaseSearch(
            environment=self._env,
            start_pos=self._start,
            seed=self._seed,
            multi_objective=self._multi_objective,
            max_pareto_path_archive=self._max_pareto,
        )
        all_values: list[dict] = []
        for positions in paths_positions:
            path_moves = self._extract_moves(positions)
            if not path_moves:
                continue
            all_values.extend(
                orchestrator.run_phase2(
                    path_moves,
                    phase2_budget_per_path,
                    per_sim_budget,
                    simulations_per_child,
                )
            )

        if not all_values:
            return []

        class _W:
            def __init__(self, v):
                self._values = v
                self._ucb_values = v
        wrapped = [_W(v) for v in all_values]
        front = Helper.determine_pareto_front_from_nodes(wrapped)
        return [w._values for w in front]
