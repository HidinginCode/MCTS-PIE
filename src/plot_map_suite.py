"""Plot the full map suite as a 5x3 matrix of heatmaps.

Rows: map types (easy, checkerboard, random, meandering_river, random_maze).
Cols: grid sizes (20, 35, 50).

Each cell shows the weight grid as a heatmap with start (white circle), goal
(gold star), and ``N_CHECKPOINTS`` deterministically sampled checkpoints
numbered in visit order.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from environment import Environment


MAP_TYPES = [
    "easy_map",
    "checkerboard_map",
    "random_map",
    "meandering_river_map",
    "random_maze",
]
ENV_DIMS = [20, 35, 50]
SEED = 1000
N_CHECKPOINTS = 5


def _load(map_type: str, env_dim: int) -> tuple[np.ndarray, list[tuple], tuple, tuple]:
    start = (0, 0)
    goal = (env_dim - 1, env_dim - 1)
    env = Environment(env_dim=env_dim, goal=goal, map_type=map_type, start_pos=start)
    cps = env.sample_checkpoints(N_CHECKPOINTS, SEED) + [goal]
    grid = np.array(env._environment, dtype=float).T  # transpose for imshow x=col, y=row
    return grid, cps, start, goal


def main() -> None:
    fig, axes = plt.subplots(
        len(MAP_TYPES), len(ENV_DIMS),
        figsize=(11, 17),
        gridspec_kw={"hspace": 0.18, "wspace": 0.18},
    )

    vmin, vmax = 0.0, 1.0  # weights normalised to [0,1]; clip outliers
    last_im = None
    for r, map_type in enumerate(MAP_TYPES):
        for c, env_dim in enumerate(ENV_DIMS):
            ax = axes[r, c]
            grid, cps, start, goal = _load(map_type, env_dim)
            im = ax.imshow(
                grid, cmap="viridis", origin="lower",
                extent=(-0.5, env_dim - 0.5, -0.5, env_dim - 0.5),
                vmin=vmin, vmax=vmax, interpolation="nearest",
            )
            last_im = im

            # Start.
            ax.scatter([start[0]], [start[1]], marker="o", s=70,
                       facecolor="white", edgecolor="black", zorder=5)
            # Checkpoints in order (excluding goal as the last, which we mark
            # separately as a star).
            for i, cp in enumerate(cps[:-1], start=1):
                ax.scatter([cp[0]], [cp[1]], marker="s", s=60,
                           facecolor="#ff7f0e", edgecolor="black", zorder=5)
                ax.annotate(str(i), xy=cp, xytext=(0, 0), textcoords="offset points",
                            ha="center", va="center", fontsize=7,
                            color="black", zorder=6,
                            fontweight="bold")
            # Goal (final checkpoint).
            ax.scatter([goal[0]], [goal[1]], marker="*", s=160,
                       facecolor="gold", edgecolor="black", zorder=5)
            ax.annotate(str(len(cps)), xy=goal, xytext=(0, 0),
                        textcoords="offset points",
                        ha="center", va="center", fontsize=7,
                        color="black", zorder=6, fontweight="bold")

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(-0.5, env_dim - 0.5)
            ax.set_ylim(-0.5, env_dim - 0.5)
            ax.set_aspect("equal")

            if r == 0:
                ax.set_title(f"{env_dim}×{env_dim}", fontsize=11)
            if c == 0:
                ax.set_ylabel(map_type.replace("_map", "").replace("_", " "),
                              fontsize=10, rotation=90, labelpad=10)

    # Shared colorbar on the right.
    cbar_ax = fig.add_axes([0.92, 0.15, 0.018, 0.7])
    fig.colorbar(last_im, cax=cbar_ax, label="cell weight")

    # Legend at the top.
    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor="white",
                   markeredgecolor="black", markersize=8, label="start"),
        plt.Line2D([0], [0], marker="s", linestyle="", markerfacecolor="#ff7f0e",
                   markeredgecolor="black", markersize=8,
                   label="checkpoints (1…N, sampled with seed=1000)"),
        plt.Line2D([0], [0], marker="*", linestyle="", markerfacecolor="gold",
                   markeredgecolor="black", markersize=12, label="goal"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 0.99), frameon=False, fontsize=9)

    fig.suptitle(
        f"MCTS-PIE map suite – 5 types × 3 sizes "
        f"(seed={SEED}, n_checkpoints={N_CHECKPOINTS})",
        y=1.005, fontsize=12,
    )

    out_dir = "./results/map_suite"
    os.makedirs(out_dir, exist_ok=True)
    base = "map_suite_5x3"
    fig.savefig(os.path.join(out_dir, base + ".png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, base + ".pdf"), bbox_inches="tight")
    print(f"Saved -> {out_dir}/{base}.png")


if __name__ == "__main__":
    main()
