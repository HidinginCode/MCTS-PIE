"""Visualize Pareto-optimal paths on the meandering_river 35x35 map.

Produces a side-by-side figure:
  - Left: the map (greyscale weights), with every unique A* movement path
    overlaid, colored by its step count.
  - Right: (step_count, weight_shifted) Pareto fronts for `astar` and
    `astar_mcts`, with each marker labelled by its step count.

The hybrid replays A*'s movement skeleton, so the *paths* are identical
between the two algorithms; only the weight-shift values differ.
"""

from __future__ import annotations

import os
import random
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from astar import A_Star
from astar_mcts_hybrid import AStarMctsHybrid


MAP_TYPE = "meandering_river_map"
ENV_DIM = 35
START = (0, 0)
GOAL = (ENV_DIM - 1, ENV_DIM - 1)
SEED = 1000
PHASE2_BUDGET_PER_PATH = 4000  # E1 default per path


def collect_astar_fronts() -> tuple[list[tuple], list[tuple], list[list[tuple]]]:
    """Return (astar_pareto, astar_paths, all_unique_paths).

    Each pareto entry is (steps, weight, path_positions); all_unique_paths is
    the deduplicated list of A* paths used by both algorithms.
    """
    solver = A_Star(MAP_TYPE, ENV_DIM, START, GOAL)
    L_star = solver.compute_shortest_steps()
    max_epsilon = ENV_DIM * 3
    seen, raw = set(), []
    for eps in range(max_epsilon + 1):
        node = solver.minimize_weight_with_step_bound(L_star + eps)
        if node is None:
            continue
        pos = tuple(solver.reconstruct_path(node))
        if pos in seen:
            continue
        seen.add(pos)
        raw.append((node.steps, float(node.weight_sum), list(pos)))

    # Pareto filter on (steps, weight)
    raw_sorted = sorted(raw, key=lambda r: (r[0], r[1]))
    pareto: list[tuple] = []
    best_w = float("inf")
    for s, w, p in raw_sorted:
        if w < best_w:
            pareto.append((s, w, p))
            best_w = w
    return pareto, [r[2] for r in pareto], [r[2] for r in raw]


def run_hybrid_pareto() -> list[dict]:
    random.seed(SEED)
    np.random.seed(SEED)
    h = AStarMctsHybrid(MAP_TYPE, ENV_DIM, START, GOAL,
                        seed=SEED, multi_objective=True,
                        max_pareto_path_archive=20)
    front = h.run(phase2_budget_per_path=PHASE2_BUDGET_PER_PATH,
                  per_sim_budget=1, simulations_per_child=1)
    # Filter to solved
    solved = [v for v in front if v["distance_to_goal"] <= 1e-9]
    # Pareto on (steps, weight)
    solved.sort(key=lambda v: (v["step_count"], v["weight_shifted"]))
    out = []
    best_w = float("inf")
    for v in solved:
        if v["weight_shifted"] < best_w:
            out.append(v)
            best_w = v["weight_shifted"]
    return out


def main() -> None:
    pareto_astar, astar_paths, all_paths = collect_astar_fronts()
    print(f"A* epsilon-sweep produced {len(all_paths)} unique paths; "
          f"{len(pareto_astar)} are Pareto-optimal on (steps, weight).")
    hybrid_front = run_hybrid_pareto()

    # Background map (weights).
    solver = A_Star(MAP_TYPE, ENV_DIM, START, GOAL)
    weight_map = np.array(solver.env._environment, dtype=float).T  # transpose -> imshow x=col,y=row

    fig, (ax_map, ax_pf) = plt.subplots(1, 2, figsize=(13, 6))

    # ---- map ----
    ax_map.imshow(weight_map, cmap="Greys", origin="lower",
                  extent=(-0.5, ENV_DIM - 0.5, -0.5, ENV_DIM - 0.5),
                  alpha=0.85)
    step_counts = sorted({len(p) - 1 for p in astar_paths})
    cmap = cm.get_cmap("viridis", max(2, len(step_counts)))
    color_for = {s: cmap(i / max(1, len(step_counts) - 1))
                 for i, s in enumerate(step_counts)}
    for path in astar_paths:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        steps = len(path) - 1
        ax_map.plot(xs, ys, "-", color=color_for[steps], linewidth=1.6,
                    alpha=0.85, label=f"{steps} steps")
    # Deduplicate legend entries.
    handles, labels = ax_map.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax_map.legend(seen.values(), seen.keys(), loc="lower right",
                  fontsize=8, framealpha=0.9, title="path length")
    ax_map.scatter([START[0]], [START[1]], marker="o", s=80,
                   facecolor="white", edgecolor="black", zorder=5, label="start")
    ax_map.scatter([GOAL[0]], [GOAL[1]], marker="*", s=160,
                   facecolor="gold", edgecolor="black", zorder=5, label="goal")
    ax_map.set_title(f"Pareto paths on {MAP_TYPE} ({ENV_DIM}x{ENV_DIM})")
    ax_map.set_xlabel("x")
    ax_map.set_ylabel("y")
    ax_map.set_xlim(-0.5, ENV_DIM - 0.5)
    ax_map.set_ylim(-0.5, ENV_DIM - 0.5)
    ax_map.set_aspect("equal")

    # ---- Pareto-front scatter ----
    a_steps = [s for s, _, _ in pareto_astar]
    a_weights = [w for _, w, _ in pareto_astar]
    h_steps = [v["step_count"] for v in hybrid_front]
    h_weights = [v["weight_shifted"] for v in hybrid_front]

    ax_pf.plot(a_steps, a_weights, "s-", color="#1f77b4", markersize=7,
               linewidth=1.4, label="astar")
    ax_pf.plot(h_steps, h_weights, "o--", color="#8c564b", markersize=7,
               linewidth=1.4, label="astar_mcts (hybrid)")
    ax_pf.set_xlabel("step_count")
    ax_pf.set_ylabel("weight_shifted")
    ax_pf.set_title("Pareto fronts (lower-left is better)")
    ax_pf.grid(True, alpha=0.3)
    ax_pf.legend()

    out_dir = "./results/pareto_paths"
    os.makedirs(out_dir, exist_ok=True)
    base = f"pareto_paths_{MAP_TYPE}_{ENV_DIM}_seed{SEED}"
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, base + ".png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, base + ".pdf"), bbox_inches="tight")
    print(f"Saved -> {out_dir}/{base}.png")
    print(f"\nastar Pareto front ({len(pareto_astar)} points):")
    for s, w, _ in pareto_astar:
        print(f"  steps={s:3d}  weight={w:.3f}")
    print(f"\nastar_mcts Pareto front ({len(hybrid_front)} points):")
    for v in hybrid_front:
        print(f"  steps={int(v['step_count']):3d}  weight={v['weight_shifted']:.3f}")


if __name__ == "__main__":
    main()
