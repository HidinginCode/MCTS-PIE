"""Unified algorithm dispatch for the experiment harness.

Each function takes a ``RunSpec`` and returns ``(front, info)`` where
``front`` is a list of objective dicts and ``info`` is a dict with
``sim_count``, ``wall_seconds``, ``peak_rss_mb``.

The functions pin both ``random.seed`` and ``numpy.random.seed`` so that
top-level RNG calls in rollouts are deterministic per ``seed``.
"""

from __future__ import annotations

import os
import random
import time
from typing import Any, Callable

import numpy as np

try:
    import psutil  # type: ignore
    _proc = psutil.Process(os.getpid())
except Exception:  # noqa: BLE001
    psutil = None
    _proc = None

from environment import Environment
from controller import Controller
from node import Node
from mc_tree import MctsTree
from mo_mc_tree import MOMctsTree
from astar import A_Star
from astar_mcts_hybrid import AStarMctsHybrid
from two_phase_mcts import TwoPhaseSearch

from experiments.config import RunSpec


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _apply_node_overrides(overrides: dict[str, Any]) -> None:
    if "oscillation_guard" in overrides:
        from node import Node
        Node.OSCILLATION_GUARD = bool(overrides["oscillation_guard"])


def _peak_rss_mb() -> float:
    if _proc is None:
        return 0.0
    try:
        return float(_proc.memory_info().rss) / (1024 * 1024)
    except Exception:  # noqa: BLE001
        return 0.0


def _collect_terminal_values(tree: MctsTree) -> list[dict]:
    """Walk the tree and collect terminal nodes' real objective vectors."""
    out, seen, stack = [], set(), [tree._root]
    while stack:
        n = stack.pop()
        if n.is_terminal_state():
            v = n._real_values
            key = (round(v["step_count"], 6),
                   round(v["weight_shifted"], 6),
                   round(v["distance_to_goal"], 6))
            if key not in seen:
                seen.add(key)
                out.append({
                    "step_count": float(v["step_count"]),
                    "weight_shifted": float(v["weight_shifted"]),
                    "distance_to_goal": float(v["distance_to_goal"]),
                })
        stack.extend(n._children.values())
    return out


def _build_env(spec: RunSpec) -> Environment:
    env = Environment(
        env_dim=spec.env_dim,
        goal=tuple(spec.goal),
        map_type=spec.map_type,
        start_pos=tuple(spec.start),
    )
    if spec.n_checkpoints > 0:
        env._checkpoints = (
            env.sample_checkpoints(spec.n_checkpoints, spec.seed)
            + [tuple(spec.goal)]
        )
    return env


def run_mcts(spec: RunSpec, **overrides: Any) -> list[dict]:
    _seed_all(spec.seed)
    env = _build_env(spec)
    ctrl = Controller(env, start_pos=tuple(spec.start))
    root = Node(controller=ctrl)
    max_solutions = overrides.get("max_solutions", 20)
    pw_c = overrides.get("pw_c", 1.5)
    pw_alpha = overrides.get("pw_alpha", 0.7)
    tree = MctsTree(root=root, seed=spec.seed, max_solutions=max_solutions,
                    progressive_widening_c=pw_c,
                    progressive_widening_alpha=pw_alpha)
    tree.search(total_budget=spec.total_budget,
                per_sim_budget=spec.per_sim_budget,
                simulations_per_child=spec.sims_per_child,
                rollout_func=overrides.get("rollout", 2),
                root_selection=overrides.get("root_selection", 0),
                tree_selection=overrides.get("tree_selection", 1))
    return _collect_terminal_values(tree)


def run_momcts(spec: RunSpec, **overrides: Any) -> list[dict]:
    _seed_all(spec.seed)
    env = _build_env(spec)
    ctrl = Controller(env, start_pos=tuple(spec.start))
    root = Node(controller=ctrl)
    max_solutions = overrides.get("max_solutions", 20)
    pw_c = overrides.get("pw_c", 1.5)
    pw_alpha = overrides.get("pw_alpha", 0.7)
    tree = MOMctsTree(root=root, seed=spec.seed, max_solutions=max_solutions,
                      progressive_widening_c=pw_c,
                      progressive_widening_alpha=pw_alpha)
    tree.search(total_budget=spec.total_budget,
                per_sim_budget=spec.per_sim_budget,
                simulations_per_child=spec.sims_per_child,
                rollout_func=overrides.get("rollout", 2),
                root_selection=overrides.get("root_selection", 0),
                tree_selection=overrides.get("tree_selection", 1))
    return tree.global_pareto_values()


def run_two_phase(spec: RunSpec, multi_objective: bool, **overrides: Any) -> list[dict]:
    _seed_all(spec.seed)
    env = _build_env(spec)
    phase_split = overrides.get("phase1_fraction", 0.75)
    phase1 = int(spec.total_budget * phase_split)
    phase2_per_path = max(1, int(spec.total_budget * (1.0 - phase_split) / 2))
    tp = TwoPhaseSearch(env, start_pos=tuple(spec.start), seed=spec.seed,
                        multi_objective=multi_objective,
                        max_pareto_path_archive=overrides.get("max_solutions", 20))
    # MO variant uses HV-based tree selection in phase 1 so the
    # multi-objective tree actually drives different exploration.
    from strategies.tree_selection import HVSelection, UCBSelection
    phase1_tree_sel = HVSelection() if multi_objective else UCBSelection()
    paths = tp.run_phase1(phase1, spec.per_sim_budget, spec.sims_per_child,
                          tree_selection=phase1_tree_sel)
    all_values: list[dict] = []
    for path in paths:
        all_values.extend(tp.run_phase2(path, phase2_per_path,
                                        spec.per_sim_budget,
                                        spec.sims_per_child))
    if not all_values:
        return []
    from helper import Helper
    class _W:
        def __init__(self, v): self._values = v; self._ucb_values = v
    wrapped = [_W(v) for v in all_values]
    return [w._values for w in Helper.determine_pareto_front_from_nodes(wrapped)]


def _checkpoint_list(spec: RunSpec) -> list | None:
    if spec.n_checkpoints <= 0:
        return None
    env = _build_env(spec)
    return list(env._checkpoints)


def run_hybrid(spec: RunSpec, **overrides: Any) -> list[dict]:
    _seed_all(spec.seed)
    h = AStarMctsHybrid(spec.map_type, spec.env_dim,
                        tuple(spec.start), tuple(spec.goal),
                        checkpoints=_checkpoint_list(spec),
                        seed=spec.seed, multi_objective=True,
                        max_pareto_path_archive=overrides.get("max_solutions", 20))
    phase2_per_path = max(1, int(spec.total_budget / 4))
    return h.run(phase2_budget_per_path=phase2_per_path,
                 per_sim_budget=spec.per_sim_budget,
                 simulations_per_child=spec.sims_per_child)


def run_astar(spec: RunSpec, **_: Any) -> list[dict]:
    _seed_all(spec.seed)
    solver = A_Star(spec.map_type, spec.env_dim,
                    tuple(spec.start), tuple(spec.goal),
                    checkpoints=_checkpoint_list(spec))
    L_star = solver.compute_shortest_steps()
    max_epsilon = solver.env_dim * 3
    out, seen = [], set()
    for epsilon in range(max_epsilon + 1):
        node = solver.minimize_weight_with_step_bound(L_star + epsilon)
        if node is None:
            continue
        key = (node.steps, round(node.weight_sum, 6))
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "step_count": float(node.steps),
            "weight_shifted": float(node.weight_sum),
            "distance_to_goal": 0.0,
        })
    return out


_DISPATCH: dict[str, Callable[..., list[dict]]] = {
    "mcts":             run_mcts,
    "mo_mcts":          run_momcts,
    "two_phase_mcts":   lambda s, **kw: run_two_phase(s, multi_objective=False, **kw),
    "two_phase_momcts": lambda s, **kw: run_two_phase(s, multi_objective=True, **kw),
    "astar_mcts":       run_hybrid,
    "astar":            run_astar,
}


def execute(spec: RunSpec) -> tuple[list[dict], dict]:
    """Run ``spec`` and return ``(front, info)``."""
    runner = _DISPATCH[spec.algo]
    t0 = time.time()
    rss_before = _peak_rss_mb()
    overrides = dict(spec.overrides or {})
    _apply_node_overrides(overrides)
    # Strip non-runner keys before forwarding.
    for k in ("label", "budget_axis", "oscillation_guard"):
        overrides.pop(k, None)
    try:
        front = runner(spec, **overrides) or []
    except Exception as exc:  # noqa: BLE001
        return [], {
            "wall_seconds": time.time() - t0,
            "peak_rss_mb": _peak_rss_mb(),
            "sim_count": 0,
            "error": repr(exc),
        }
    rss_after = _peak_rss_mb()
    info = {
        "wall_seconds": time.time() - t0,
        "peak_rss_mb": max(rss_before, rss_after),
        "sim_count": spec.total_budget,
        "error": "",
    }
    return front, info
