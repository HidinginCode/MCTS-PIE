"""E3 ablation — Pareto archive cap (max_solutions)."""

from __future__ import annotations

import argparse

from experiments.config import MAP_SUITE, SEEDS, RunSpec
from experiments.runner import run

OUT_DIR = "./log/e3_ablations/archive_cap"
FAMILY = "e3_archive_cap"

CAPS = [10, 20, 50, 100]


def build_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    for map_type, env_dim in MAP_SUITE:
        for seed in SEEDS:
            for algo in ("mcts", "mo_mcts"):
                for cap in CAPS:
                    specs.append(RunSpec(
                        family=FAMILY, algo=algo,
                        map_type=map_type, env_dim=env_dim, seed=seed,
                        overrides={"max_solutions": cap, "label": f"cap{cap}"},
                    ))
    return specs


def main(n_workers: int = 1, resume: bool = True) -> None:
    run(FAMILY, build_specs(), OUT_DIR, n_workers=n_workers, resume=resume)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()
    main(n_workers=args.workers, resume=not args.no_resume)
