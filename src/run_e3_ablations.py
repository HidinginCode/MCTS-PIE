"""E3 — runs every ablation axis sequentially."""

from __future__ import annotations

import argparse

import run_e3_archive_cap
import run_e3_oscillation_guard
import run_e3_phase_split
import run_e3_progressive_widening
import run_e3_rollout
import run_e3_root_selection
import run_e3_tree_selection

AXES = [
    run_e3_tree_selection,
    run_e3_root_selection,
    run_e3_rollout,
    run_e3_progressive_widening,
    run_e3_phase_split,
    run_e3_archive_cap,
    run_e3_oscillation_guard,
]


def main(n_workers: int = 1, resume: bool = True) -> None:
    for mod in AXES:
        print(f"=== {mod.__name__} ===")
        mod.main(n_workers=n_workers, resume=resume)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()
    main(n_workers=args.workers, resume=not args.no_resume)
