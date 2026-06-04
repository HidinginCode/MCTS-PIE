"""Plots and tables for E1 — main comparison."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.plotting import (
    ALGO_COLORS, ALGO_ORDER, load_archive, load_runs,
    per_run_metrics, save_fig,
)
from experiments.stats import (
    critical_difference_diagram, friedman, pairwise_wilcoxon_holm,
    ranks_per_block, vargha_delaney,
)

FAMILY_DIR = "./log/e1_main"
OUT_DIR = "./results/e1_main"


def _scatter_pareto(runs: pd.DataFrame, out_dir: str) -> None:
    keys = sorted(set(zip(runs["map_type"], runs["env_dim"])))
    cols = min(3, len(keys))
    rows = (len(keys) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.5 * rows),
                             squeeze=False)
    for ax, key in zip(axes.flat, keys):
        sub = runs[(runs["map_type"] == key[0]) & (runs["env_dim"] == key[1])]
        for algo in ALGO_ORDER:
            xs, ys = [], []
            for _, r in sub[sub["algo"] == algo].iterrows():
                for p in load_archive(FAMILY_DIR, r["run_id"]):
                    if p["distance_to_goal"] <= 1e-9:
                        xs.append(p["step_count"])
                        ys.append(p["weight_shifted"])
            if xs:
                ax.scatter(xs, ys, label=algo, color=ALGO_COLORS.get(algo),
                           s=14, alpha=0.5, edgecolors="none")
        ax.set_title(f"{key[0]} ({key[1]}x{key[1]})")
        ax.set_xlabel("step count")
        ax.set_ylabel("weight shifted")
        ax.grid(True, alpha=0.3)
    for ax in axes.flat[len(keys):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(ALGO_ORDER),
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("E1 Pareto fronts (solved, lower-left is better)")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    save_fig(fig, "pareto_fronts_by_map", out_dir)
    plt.close(fig)


def _bar_metric(metrics: pd.DataFrame, metric: str, out_dir: str,
                higher_is_better: bool) -> None:
    keys = sorted(set(zip(metrics["map_type"], metrics["env_dim"])))
    fig, ax = plt.subplots(figsize=(1.1 * len(keys) * len(ALGO_ORDER) + 2, 4.5))
    width = 0.85 / len(ALGO_ORDER)
    x = np.arange(len(keys))
    for i, algo in enumerate(ALGO_ORDER):
        means, errs = [], []
        for k in keys:
            vals = metrics[(metrics["algo"] == algo)
                           & (metrics["map_type"] == k[0])
                           & (metrics["env_dim"] == k[1])][metric].dropna()
            means.append(float(vals.mean()) if len(vals) else 0.0)
            errs.append(float(vals.std()) if len(vals) else 0.0)
        ax.bar(x + i * width, means, width, yerr=errs, capsize=3,
               label=algo, color=ALGO_COLORS.get(algo))
    ax.set_xticks(x + width * (len(ALGO_ORDER) - 1) / 2)
    ax.set_xticklabels([f"{k[0]}\nd{k[1]}" for k in keys], rotation=20, ha="right")
    direction = "higher is better" if higher_is_better else "lower is better"
    ax.set_ylabel(f"{metric} ({direction})")
    ax.set_title(f"E1 {metric} per (map, dim) — mean ± std over seeds")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, f"{metric}_by_algo", out_dir)
    plt.close(fig)


def _runtime(runs: pd.DataFrame, out_dir: str) -> None:
    g = (runs.groupby(["algo", "map_type", "env_dim"])["wall_seconds"]
            .mean().reset_index())
    pivot = g.pivot_table(index="algo",
                          columns=["map_type", "env_dim"],
                          values="wall_seconds")
    pivot = pivot.reindex(ALGO_ORDER)
    fig, ax = plt.subplots(figsize=(1.2 * len(pivot.columns) + 2, 4.5))
    pivot.plot(kind="bar", ax=ax, width=0.85,
               color=[ALGO_COLORS.get(a, "#777") for a in pivot.index])
    ax.set_ylabel("seconds (mean over seeds)")
    ax.set_title("E1 wall-clock runtime per algorithm / map")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "runtime_by_algo", out_dir)
    plt.close(fig)


def _stats_and_cd(metrics: pd.DataFrame, out_dir: str) -> None:
    """Friedman + Wilcoxon-Holm + CD diagram on hypervolume."""
    pivot = (metrics.pivot_table(index=["map_type", "env_dim", "seed"],
                                  columns="algo", values="hv")
                    .reindex(columns=ALGO_ORDER).dropna())
    if pivot.empty:
        print("E1 stats: no data after pivot")
        return
    matrix = pivot.to_numpy(dtype=float)
    chi2, p = friedman(matrix)
    wh = pairwise_wilcoxon_holm(matrix, ALGO_ORDER)
    a12_rows = []
    for i, a in enumerate(ALGO_ORDER):
        for j, b in enumerate(ALGO_ORDER):
            if i >= j:
                continue
            a12, mag = vargha_delaney(matrix[:, i], matrix[:, j])
            a12_rows.append({"a": a, "b": b, "a12": a12, "magnitude": mag})
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame([{"chi2": chi2, "p": p}]).to_csv(
        os.path.join(out_dir, "friedman_hv.csv"), index=False)
    wh.to_csv(os.path.join(out_dir, "wilcoxon_holm_hv.csv"), index=False)
    pd.DataFrame(a12_rows).to_csv(
        os.path.join(out_dir, "vargha_delaney_hv.csv"), index=False)

    rank_matrix = ranks_per_block(matrix, lower_is_better=False)
    critical_difference_diagram(
        rank_matrix, ALGO_ORDER,
        savepath=os.path.join(out_dir, "cd_diagram_hv"),
        title=f"E1 critical-difference (HV)  Friedman p={p:.2e}",
    )


def main() -> None:
    runs = load_runs(FAMILY_DIR)
    if runs.empty:
        raise SystemExit("E1 runs.csv empty")
    metrics = per_run_metrics(FAMILY_DIR, runs)
    metrics.to_csv(os.path.join(FAMILY_DIR, "metrics.csv"), index=False)
    _scatter_pareto(runs, OUT_DIR)
    _bar_metric(metrics, "hv", OUT_DIR, higher_is_better=True)
    _bar_metric(metrics, "igd_plus", OUT_DIR, higher_is_better=False)
    _bar_metric(metrics, "epsilon", OUT_DIR, higher_is_better=False)
    _runtime(runs, OUT_DIR)
    _stats_and_cd(metrics, OUT_DIR)
    print(f"E1 plots/tables -> {OUT_DIR}")


if __name__ == "__main__":
    main()
