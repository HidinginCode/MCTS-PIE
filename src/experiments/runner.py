"""Generic resumable runner over a list of ``RunSpec``.

Per-run outputs:
    <out_dir>/archives/<run_id>.csv     - one row per Pareto-front point
    <out_dir>/runs.csv                  - one row per run, aggregated metrics
    <out_dir>/manifest.json             - completed run_ids
    <out_dir>/provenance.json           - written once at start
"""

from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import time
from typing import Iterable

import pandas as pd

from experiments.algorithms import execute
from experiments.config import RunSpec
from experiments.metrics import OBJECTIVES
from experiments.provenance import snapshot


RUN_COLUMNS = [
    "family", "algo", "map_type", "env_dim", "n_checkpoints",
    "total_budget", "per_sim_budget", "sims_per_child", "seed",
    "overrides", "n_points", "wall_seconds", "peak_rss_mb",
    "sim_count", "error", "run_id",
]


def _archive_path(out_dir: str, run_id: str) -> str:
    return os.path.join(out_dir, "archives", f"{run_id}.csv")


def _runs_csv_path(out_dir: str) -> str:
    return os.path.join(out_dir, "runs.csv")


def _manifest_path(out_dir: str) -> str:
    return os.path.join(out_dir, "manifest.json")


def _load_manifest(out_dir: str) -> set[str]:
    p = _manifest_path(out_dir)
    if not os.path.exists(p):
        return set()
    with open(p, "r", encoding="utf-8") as f:
        return set(json.load(f))


def _save_manifest(out_dir: str, done: set[str]) -> None:
    with open(_manifest_path(out_dir), "w", encoding="utf-8") as f:
        json.dump(sorted(done), f, indent=2)


def _write_archive(path: str, front: list[dict], spec: RunSpec) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["point_idx", *OBJECTIVES])
        for i, v in enumerate(front):
            w.writerow([i] + [float(v[o]) for o in OBJECTIVES])


def _append_run_row(path: str, row: dict) -> None:
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RUN_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


def _execute_one(spec: RunSpec) -> tuple[RunSpec, list[dict], dict]:
    front, info = execute(spec)
    return spec, front, info


def run(family: str, specs: Iterable[RunSpec], out_dir: str,
        n_workers: int = 1, resume: bool = True) -> None:
    os.makedirs(os.path.join(out_dir, "archives"), exist_ok=True)
    if not os.path.exists(os.path.join(out_dir, "provenance.json")):
        snapshot(out_dir, extra={"family": family})

    done = _load_manifest(out_dir) if resume else set()
    pending = [s for s in specs if not (resume and s.run_id in done)]
    if not pending:
        print(f"[{family}] nothing to do (manifest has {len(done)} runs)")
        return

    print(f"[{family}] {len(pending)} runs pending, {n_workers} workers")
    runs_csv = _runs_csv_path(out_dir)

    def handle(result: tuple[RunSpec, list[dict], dict]) -> None:
        spec, front, info = result
        _write_archive(_archive_path(out_dir, spec.run_id), front, spec)
        _append_run_row(runs_csv, {
            "family": spec.family,
            "algo": spec.algo,
            "map_type": spec.map_type,
            "env_dim": spec.env_dim,
            "n_checkpoints": spec.n_checkpoints,
            "total_budget": spec.total_budget,
            "per_sim_budget": spec.per_sim_budget,
            "sims_per_child": spec.sims_per_child,
            "seed": spec.seed,
            "overrides": json.dumps(spec.overrides, sort_keys=True),
            "n_points": len(front),
            "wall_seconds": info.get("wall_seconds", 0.0),
            "peak_rss_mb": info.get("peak_rss_mb", 0.0),
            "sim_count": info.get("sim_count", 0),
            "error": info.get("error", ""),
            "run_id": spec.run_id,
        })
        done.add(spec.run_id)
        _save_manifest(out_dir, done)
        n = len(done)
        print(f"  [{n}] {spec.run_id}  pts={len(front)}  "
              f"t={info.get('wall_seconds', 0):.1f}s  "
              f"err={info.get('error', '')[:40]}", flush=True)

    if n_workers <= 1:
        for s in pending:
            handle(_execute_one(s))
        return

    # Spawn-context pool — Windows-safe.
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        for result in pool.imap_unordered(_execute_one, pending):
            handle(result)
