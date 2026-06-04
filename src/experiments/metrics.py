"""Multi-objective performance indicators for the paper protocol.

Conventions
-----------
- All objectives are minimized: (step_count, weight_shifted, distance_to_goal).
- Reference points for HV are *per-map*, computed from the union front of all
  algorithms across all seeds, scaled by 1.05 plus a +1 slack.
- Reference fronts for IGD+/epsilon are the non-dominated union of every
  algorithm's archive on a given map; persisted as
  ``./log/<family>/reference_front_<map>.csv``.

`pymoo` is required. With three objectives the exact HV in pymoo is fine;
if the problem grows we should swap to ``hv_approx``.
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from pymoo.indicators.hv import HV  # type: ignore
    from pymoo.indicators.igd_plus import IGDPlus  # type: ignore
except Exception as exc:  # noqa: BLE001
    raise ImportError("pymoo is required for experiments.metrics") from exc


OBJECTIVES = ("step_count", "weight_shifted", "distance_to_goal")


def points_array(points: Iterable[dict]) -> np.ndarray:
    arr = np.asarray([[p[o] for o in OBJECTIVES] for p in points], dtype=float)
    return arr.reshape(-1, len(OBJECTIVES))


def is_non_dominated(arr: np.ndarray) -> np.ndarray:
    n = arr.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        # j dominates i if j <= i on all and < on any.
        dom = np.all(arr <= arr[i], axis=1) & np.any(arr < arr[i], axis=1)
        dom[i] = False
        if np.any(dom):
            keep[i] = False
    return keep


def non_dominated_union(point_lists: Iterable[Iterable[dict]]) -> np.ndarray:
    pts = []
    for lst in point_lists:
        if lst:
            pts.extend(lst)
    if not pts:
        return np.empty((0, len(OBJECTIVES)))
    arr = points_array(pts)
    return arr[is_non_dominated(arr)]


def reference_point(arr: np.ndarray, slack: float = 1.05, pad: float = 1.0) -> np.ndarray:
    if arr.size == 0:
        return np.array([1.0] * len(OBJECTIVES))
    return arr.max(axis=0) * slack + pad


def hv(points: list[dict], ref: np.ndarray) -> float:
    if not points:
        return 0.0
    return float(HV(ref_point=ref)(points_array(points)))


def igd_plus(points: list[dict], reference_front: np.ndarray) -> float:
    if not points or reference_front.size == 0:
        return float("nan")
    return float(IGDPlus(reference_front)(points_array(points)))


def epsilon_indicator(points: list[dict], reference_front: np.ndarray) -> float:
    """Additive epsilon-indicator: max over ref of min over approx of max-coord diff."""
    if not points or reference_front.size == 0:
        return float("nan")
    A = points_array(points)
    R = reference_front
    eps = -np.inf
    for r in R:
        diffs = np.max(A - r, axis=1)
        eps = max(eps, float(np.min(diffs)))
    return eps


def normalize_to_unit(arr: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    span = np.where(hi > lo, hi - lo, 1.0)
    return (arr - lo) / span


def build_reference_front(
    archives: dict[str, list[dict]], save_path: str | None = None
) -> np.ndarray:
    """Build a per-map reference front from every algorithm's solved points.

    Parameters
    ----------
    archives: mapping from arbitrary key (e.g. "{algo}|{seed}") to its Pareto archive.
    """
    front = non_dominated_union(archives.values())
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pd.DataFrame(front, columns=list(OBJECTIVES)).to_csv(save_path, index=False)
    return front


def load_reference_front(path: str) -> np.ndarray:
    if not os.path.exists(path):
        return np.empty((0, len(OBJECTIVES)))
    df = pd.read_csv(path)
    return df[list(OBJECTIVES)].to_numpy(dtype=float)


def assert_dominates(reference: np.ndarray, candidates: np.ndarray) -> None:
    """Sanity check: every candidate must be weakly dominated by some ref point."""
    if reference.size == 0:
        return
    for c in candidates:
        weak = np.all(reference <= c, axis=1)
        if not np.any(weak):
            raise AssertionError(
                f"Reference front does not dominate candidate {c.tolist()}"
            )
