"""Common loaders, color palette and helpers for the per-family plotters."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.metrics import (
    OBJECTIVES, build_reference_front, hv, igd_plus, epsilon_indicator,
    reference_point, load_reference_front,
)


ALGO_ORDER = [
    "astar", "mcts", "mo_mcts",
    "two_phase_mcts", "two_phase_momcts", "astar_mcts",
]
ALGO_COLORS = {
    "astar":            "#1f77b4",
    "mcts":             "#ff7f0e",
    "mo_mcts":          "#2ca02c",
    "two_phase_mcts":   "#d62728",
    "two_phase_momcts": "#9467bd",
    "astar_mcts":       "#8c564b",
}


def load_runs(family_dir: str) -> pd.DataFrame:
    p = os.path.join(family_dir, "runs.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    df["overrides"] = df["overrides"].fillna("{}").apply(json.loads)
    return df


def load_archive(family_dir: str, run_id: str) -> list[dict]:
    p = os.path.join(family_dir, "archives", f"{run_id}.csv")
    if not os.path.exists(p):
        return []
    df = pd.read_csv(p)
    return [{o: float(v) for o, v in zip(OBJECTIVES, row)}
            for row in df[list(OBJECTIVES)].to_numpy()]


def reference_fronts_per_map(family_dir: str, runs: pd.DataFrame,
                             solved_only: bool = True
                             ) -> tuple[dict, dict]:
    """Build (or load cached) per-map reference fronts and HV reference points.

    Returns ``(fronts, ref_points)``. The reference *front* is the
    non-dominated union of every archive (used for IGD+/epsilon). The
    reference *point* is anchored to the per-map *worst* observed point
    across every algorithm, so dominated solved points still contribute
    positive hypervolume.
    """
    out_dir = os.path.join(family_dir, "reference")
    os.makedirs(out_dir, exist_ok=True)
    fronts, refs_pt = {}, {}
    from experiments.metrics import points_array, reference_point
    for map_key, sub in runs.groupby(["map_type", "env_dim"]):
        cache = os.path.join(out_dir, f"ref_{map_key[0]}_d{map_key[1]}.csv")
        archives = {}
        all_points: list[dict] = []
        for _, r in sub.iterrows():
            arc = load_archive(family_dir, r["run_id"])
            if solved_only:
                arc = [p for p in arc if p["distance_to_goal"] <= 1e-9]
            archives[r["run_id"]] = arc
            all_points.extend(arc)
        if os.path.exists(cache):
            fronts[map_key] = load_reference_front(cache)
        else:
            fronts[map_key] = build_reference_front(archives, save_path=cache)
        if all_points:
            refs_pt[map_key] = reference_point(points_array(all_points))
        else:
            refs_pt[map_key] = np.array([1.0, 1.0, 1.0])
    return fronts, refs_pt


def per_run_metrics(family_dir: str, runs: pd.DataFrame,
                    solved_only: bool = True) -> pd.DataFrame:
    fronts, ref_pts = reference_fronts_per_map(family_dir, runs,
                                               solved_only=solved_only)
    rows = []
    for _, r in runs.iterrows():
        key = (r["map_type"], r["env_dim"])
        ref_front = fronts.get(key, np.empty((0, len(OBJECTIVES))))
        ref = ref_pts.get(key, np.array([1.0] * len(OBJECTIVES)))
        arc = load_archive(family_dir, r["run_id"])
        if solved_only:
            arc = [p for p in arc if p["distance_to_goal"] <= 1e-9]
        rows.append({
            **{k: r[k] for k in ("algo", "map_type", "env_dim", "seed",
                                 "total_budget", "n_checkpoints",
                                 "wall_seconds", "run_id")},
            "hv": hv(arc, ref),
            "igd_plus": igd_plus(arc, ref_front),
            "epsilon": epsilon_indicator(arc, ref_front),
            "n_points": len(arc),
        })
    return pd.DataFrame(rows)


def save_fig(fig, name: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{name}.pdf"), bbox_inches="tight")
