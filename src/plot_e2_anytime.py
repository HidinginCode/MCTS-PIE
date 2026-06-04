"""E2 anytime curves: HV vs simulation budget per algorithm/map."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from experiments.plotting import (
    ALGO_COLORS, ALGO_ORDER, load_runs, per_run_metrics, save_fig,
)
from experiments.stats import bootstrap_ci

FAMILY_DIR = "./log/e2_anytime"
OUT_DIR = "./results/e2_anytime"


def main() -> None:
    runs = load_runs(FAMILY_DIR)
    metrics = per_run_metrics(FAMILY_DIR, runs)
    metrics.to_csv(os.path.join(FAMILY_DIR, "metrics.csv"), index=False)

    keys = sorted(set(zip(metrics["map_type"], metrics["env_dim"])))
    cols = min(3, len(keys))
    rows = (len(keys) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 4 * rows),
                             squeeze=False)
    for ax, k in zip(axes.flat, keys):
        sub = metrics[(metrics["map_type"] == k[0]) & (metrics["env_dim"] == k[1])]
        for algo in ALGO_ORDER:
            d = sub[sub["algo"] == algo]
            budgets = sorted(d["total_budget"].unique())
            means, los, his = [], [], []
            for b in budgets:
                vals = d[d["total_budget"] == b]["hv"].dropna().to_numpy()
                if len(vals) == 0:
                    means.append(np.nan); los.append(np.nan); his.append(np.nan)
                    continue
                means.append(float(vals.mean()))
                lo, hi = bootstrap_ci(vals)
                los.append(lo); his.append(hi)
            ax.plot(budgets, means, "-o", label=algo,
                    color=ALGO_COLORS.get(algo), linewidth=1.5, markersize=4)
            ax.fill_between(budgets, los, his, color=ALGO_COLORS.get(algo),
                            alpha=0.15)
        ax.set_xscale("log")
        ax.set_title(f"{k[0]} ({k[1]}x{k[1]})")
        ax.set_xlabel("simulation budget")
        ax.set_ylabel("hypervolume")
        ax.grid(True, alpha=0.3, which="both")
    for ax in axes.flat[len(keys):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(ALGO_ORDER),
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("E2 anytime hypervolume (95% bootstrap CI)")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save_fig(fig, "anytime_hv", OUT_DIR)
    plt.close(fig)
    print(f"E2 plots -> {OUT_DIR}")


if __name__ == "__main__":
    main()
