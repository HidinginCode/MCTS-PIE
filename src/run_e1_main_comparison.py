"""E1 — Main comparison: 6 algorithms x 10 maps x 30 seeds at fixed budget."""

from __future__ import annotations

import argparse

from experiments.config import (
    ALGORITHMS, MAP_SUITE, SEEDS, RunSpec,
    DEFAULT_TOTAL_BUDGET,
)
from experiments.runner import run

OUT_DIR = "./log/e1_main"
FAMILY = "e1_main"


def build_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    for map_type, env_dim in MAP_SUITE:
        for seed in SEEDS:
            for algo in ALGORITHMS:
                specs.append(RunSpec(
                    family=FAMILY, algo=algo,
                    map_type=map_type, env_dim=env_dim,
                    seed=seed, total_budget=DEFAULT_TOTAL_BUDGET,
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
