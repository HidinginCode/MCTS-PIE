"""E4 — scaling plots: HV vs env_dim, HV vs n_checkpoints, runtime vs env_dim."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from experiments.plotting import (
    ALGO_COLORS, ALGO_ORDER, load_runs, per_run_metrics, save_fig,
)
from experiments.stats import bootstrap_ci

FAMILY_DIR = "./log/e4_scaling"
OUT_DIR = "./results/e4_scaling"


def _line(metrics, x_col: str, value_col: str, title: str, fname: str,
          xscale: str = "linear") -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for algo in ALGO_ORDER:
        sub = metrics[metrics["algo"] == algo]
        xs = sorted(sub[x_col].unique())
        means, los, his = [], [], []
        for x in xs:
            vals = sub[sub[x_col] == x][value_col].dropna().to_numpy()
            if len(vals) == 0:
                means.append(np.nan); los.append(np.nan); his.append(np.nan)
                continue
            means.append(float(vals.mean()))
            lo, hi = bootstrap_ci(vals)
            los.append(lo); his.append(hi)
        ax.plot(xs, means, "-o", label=algo, color=ALGO_COLORS.get(algo))
        ax.fill_between(xs, los, his, color=ALGO_COLORS.get(algo), alpha=0.15)
    ax.set_xlabel(x_col)
    ax.set_ylabel(value_col)
    if xscale == "log":
        ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, fname, OUT_DIR)
    plt.close(fig)


def main() -> None:
    runs = load_runs(FAMILY_DIR)
    metrics = per_run_metrics(FAMILY_DIR, runs)
    metrics.to_csv(os.path.join(FAMILY_DIR, "metrics.csv"), index=False)

    _line(metrics, "env_dim", "hv", "E4 HV vs env_dim", "hv_vs_env_dim")
    _line(metrics, "n_checkpoints", "hv",
          "E4 HV vs number of checkpoints", "hv_vs_n_checkpoints")
    _line(metrics, "env_dim", "wall_seconds",
          "E4 runtime vs env_dim (log-log)",
          "runtime_vs_env_dim", xscale="log")
    print(f"E4 plots -> {OUT_DIR}")


if __name__ == "__main__":
    main()
