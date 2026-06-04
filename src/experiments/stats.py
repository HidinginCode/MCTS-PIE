"""Statistical tests and a critical-difference plotter."""

from __future__ import annotations

import itertools
import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import stats as sp_stats


def friedman(matrix: np.ndarray) -> tuple[float, float]:
    """Friedman test. ``matrix`` shape (n_blocks, n_algos)."""
    if matrix.shape[1] < 3 or matrix.shape[0] < 2:
        return float("nan"), float("nan")
    chi2, p = sp_stats.friedmanchisquare(*[matrix[:, j] for j in range(matrix.shape[1])])
    return float(chi2), float(p)


def pairwise_wilcoxon_holm(
    matrix: np.ndarray, names: Sequence[str], alternative: str = "two-sided"
) -> pd.DataFrame:
    """Pairwise paired Wilcoxon signed-rank with Holm-Bonferroni correction.

    ``matrix`` shape (n_blocks, n_algos). Returns a long-form DataFrame with
    columns: a, b, statistic, p_raw, p_holm.
    """
    n_algos = matrix.shape[1]
    pairs = list(itertools.combinations(range(n_algos), 2))
    rows = []
    raw_ps = []
    for i, j in pairs:
        diff = matrix[:, i] - matrix[:, j]
        if np.allclose(diff, 0):
            stat, p = 0.0, 1.0
        else:
            try:
                stat, p = sp_stats.wilcoxon(matrix[:, i], matrix[:, j],
                                            alternative=alternative,
                                            zero_method="wilcox")
            except ValueError:
                stat, p = float("nan"), 1.0
        rows.append({"a": names[i], "b": names[j], "statistic": float(stat),
                     "p_raw": float(p)})
        raw_ps.append(float(p))

    # Holm-Bonferroni on the family of pairwise tests.
    order = np.argsort(raw_ps)
    m = len(raw_ps)
    holm = [1.0] * m
    cmax = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * raw_ps[idx]
        cmax = max(cmax, adj)
        holm[idx] = min(1.0, cmax)
    for r, h in zip(rows, holm):
        r["p_holm"] = float(h)
    return pd.DataFrame(rows)


def vargha_delaney(a: np.ndarray, b: np.ndarray) -> tuple[float, str]:
    """Vargha-Delaney A12 effect size with magnitude label."""
    a = np.asarray(a)
    b = np.asarray(b)
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0:
        return float("nan"), "n/a"
    # rank-based formulation
    combined = np.concatenate([a, b])
    ranks = sp_stats.rankdata(combined)
    r_a = ranks[:n_a].sum()
    a12 = (r_a / n_a - (n_a + 1) / 2) / n_b
    d = abs(a12 - 0.5)
    if d < 0.06:
        mag = "negligible"
    elif d < 0.14:
        mag = "small"
    elif d < 0.21:
        mag = "medium"
    else:
        mag = "large"
    return float(a12), mag


def bootstrap_ci(values: np.ndarray, n: int = 10000, alpha: float = 0.05,
                 rng: np.random.Generator | None = None) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    means = values[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def ranks_per_block(matrix: np.ndarray, lower_is_better: bool = False) -> np.ndarray:
    """Return per-block (per-row) ranks. By convention HV is *higher is better*."""
    if lower_is_better:
        return np.apply_along_axis(sp_stats.rankdata, 1, matrix)
    return np.apply_along_axis(lambda r: sp_stats.rankdata(-r), 1, matrix)


def critical_difference(n_blocks: int, n_algos: int, alpha: float = 0.05) -> float:
    """Nemenyi critical difference for a Friedman setup."""
    # Studentized range q values (Nemenyi) at alpha=0.05, for k = 2..15.
    q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
           9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354,
           15: 3.391}
    q10 = {2: 1.645, 3: 2.052, 4: 2.291, 5: 2.460, 6: 2.589, 7: 2.693, 8: 2.780,
           9: 2.855, 10: 2.920, 11: 2.978, 12: 3.030, 13: 3.077, 14: 3.120,
           15: 3.159}
    table = q05 if alpha == 0.05 else q10
    if n_algos not in table:
        # Closest available.
        n_algos = min(table, key=lambda k: abs(k - n_algos))
    q = table[n_algos]
    return float(q * np.sqrt(n_algos * (n_algos + 1) / (6.0 * n_blocks)))


def critical_difference_diagram(rank_matrix: np.ndarray, names: Sequence[str],
                                alpha: float = 0.05, savepath: str | None = None,
                                title: str = "") -> None:
    """Demsar-style CD plot. ``rank_matrix`` shape (n_blocks, n_algos)."""
    n_blocks, n_algos = rank_matrix.shape
    avg_ranks = rank_matrix.mean(axis=0)
    cd = critical_difference(n_blocks, n_algos, alpha=alpha)

    order = np.argsort(avg_ranks)
    sorted_names = [names[i] for i in order]
    sorted_ranks = avg_ranks[order]

    fig, ax = plt.subplots(figsize=(9, 2 + 0.25 * n_algos))
    ax.set_xlim(0.5, n_algos + 0.5)
    ax.set_ylim(-0.5, n_algos / 2 + 1)
    ax.invert_yaxis()
    ax.axis("off")

    # axis line
    ax.plot([1, n_algos], [0, 0], "k-")
    for tick in range(1, n_algos + 1):
        ax.plot([tick, tick], [-0.05, 0.05], "k-")
        ax.text(tick, 0.15, str(tick), ha="center", va="top", fontsize=9)

    # CD bar
    ax.plot([1, 1 + cd], [-0.4, -0.4], "k-", lw=2)
    ax.text(1 + cd / 2, -0.55, f"CD = {cd:.2f}", ha="center", fontsize=9)

    # name lines
    half = (n_algos + 1) // 2
    for rank_idx, (name, r) in enumerate(zip(sorted_names, sorted_ranks)):
        is_left = rank_idx < half
        y = (rank_idx + 1 if is_left else (n_algos - rank_idx)) * 0.4
        x_text = 0.6 if is_left else n_algos + 0.4
        ax.plot([r, r], [0, y], "k-", lw=0.8)
        ax.plot([r, x_text + (0 if is_left else 0)], [y, y], "k-", lw=0.8)
        ax.text(x_text, y, f"{name} ({r:.2f})",
                ha="right" if is_left else "left", va="center", fontsize=10)

    # cliques: connect groups whose rank gap < CD
    cliques = []
    i = 0
    while i < n_algos:
        j = i
        while j + 1 < n_algos and (sorted_ranks[j + 1] - sorted_ranks[i]) < cd:
            j += 1
        if j > i:
            cliques.append((sorted_ranks[i], sorted_ranks[j]))
        i = j + 1
    for k, (lo, hi) in enumerate(cliques):
        y = -0.2 - k * 0.08
        ax.plot([lo, hi], [y, y], "k-", lw=3)

    if title:
        ax.set_title(title, fontsize=11)
    if savepath:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath + ".png", dpi=150, bbox_inches="tight")
        fig.savefig(savepath + ".pdf", bbox_inches="tight")
    plt.close(fig)
