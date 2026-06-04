"""E4 — Environment scaling: env_dim x n_checkpoints x map_type x algos."""

from __future__ import annotations

import argparse

from experiments.config import ALGORITHMS, MAP_TYPES, SEEDS, RunSpec
from experiments.runner import run

OUT_DIR = "./log/e4_scaling"
FAMILY = "e4_scaling"

ENV_DIMS = [20, 35, 50]
N_CHECKPOINTS = [1, 3, 5, 10]
# Budget grows proportionally with env_dim.
BUDGET_BY_DIM = {20: 4000, 35: 8000, 50: 16000}


def build_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    for env_dim in ENV_DIMS:
        for n_cp in N_CHECKPOINTS:
            for map_type in MAP_TYPES:
                for seed in SEEDS:
                    for algo in ALGORITHMS:
                        specs.append(RunSpec(
                            family=FAMILY, algo=algo,
                            map_type=map_type, env_dim=env_dim,
                            seed=seed,
                            total_budget=BUDGET_BY_DIM[env_dim],
                            n_checkpoints=n_cp,
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
