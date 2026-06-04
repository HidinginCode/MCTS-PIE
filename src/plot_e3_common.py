"""Generic per-axis ablation plotter for E3."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.plotting import load_runs, per_run_metrics, save_fig
from experiments.stats import friedman, pairwise_wilcoxon_holm


def _variant_label(row: pd.Series) -> str:
    ov = row["overrides"] or {}
    label = ov.get("label", "")
    return f"{row['algo']}/{label}" if label else row["algo"]


def plot_axis(family_dir: str, out_dir: str, axis_name: str) -> None:
    runs = load_runs(family_dir)
    if runs.empty:
        print(f"[{axis_name}] runs.csv empty")
        return
    runs["variant"] = runs.apply(_variant_label, axis=1)
    metrics = per_run_metrics(family_dir, runs)
    metrics["variant"] = metrics.apply(
        lambda r: _variant_label(runs[runs["run_id"] == r["run_id"]].iloc[0]),
        axis=1,
    )
    metrics.to_csv(os.path.join(family_dir, "metrics.csv"), index=False)

    variants = sorted(metrics["variant"].unique())
    keys = sorted(set(zip(metrics["map_type"], metrics["env_dim"])))

    fig, ax = plt.subplots(figsize=(1.0 * len(keys) * len(variants) + 2, 4.5))
    width = 0.85 / max(1, len(variants))
    x = np.arange(len(keys))
    for i, v in enumerate(variants):
        means, errs = [], []
        for k in keys:
            vals = metrics[(metrics["variant"] == v)
                           & (metrics["map_type"] == k[0])
                           & (metrics["env_dim"] == k[1])]["hv"].dropna()
            means.append(float(vals.mean()) if len(vals) else 0.0)
            errs.append(float(vals.std()) if len(vals) else 0.0)
        ax.bar(x + i * width, means, width, yerr=errs, capsize=3, label=v)
    ax.set_xticks(x + width * (len(variants) - 1) / 2)
    ax.set_xticklabels([f"{k[0]}\nd{k[1]}" for k in keys], rotation=20, ha="right")
    ax.set_ylabel("hypervolume")
    ax.set_title(f"E3 {axis_name} — HV per (map, dim)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, f"{axis_name}_hv_by_variant", out_dir)
    plt.close(fig)

    pivot = (metrics.pivot_table(index=["map_type", "env_dim", "seed"],
                                  columns="variant", values="hv")
                    .dropna())
    if pivot.shape[1] >= 2 and pivot.shape[0] >= 2:
        chi2, p = friedman(pivot.to_numpy(dtype=float))
        wh = pairwise_wilcoxon_holm(pivot.to_numpy(dtype=float),
                                    list(pivot.columns))
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame([{"chi2": chi2, "p": p}]).to_csv(
            os.path.join(out_dir, f"{axis_name}_friedman_hv.csv"), index=False)
        wh.to_csv(os.path.join(out_dir, f"{axis_name}_wilcoxon_holm_hv.csv"),
                  index=False)
    print(f"E3 [{axis_name}] -> {out_dir}")
