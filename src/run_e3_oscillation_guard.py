"""E3 ablation — oscillation guard on/off."""

from __future__ import annotations

import argparse

from experiments.config import MAP_SUITE, SEEDS, RunSpec
from experiments.runner import run

OUT_DIR = "./log/e3_ablations/oscillation_guard"
FAMILY = "e3_oscillation_guard"


def build_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    for map_type, env_dim in MAP_SUITE:
        for seed in SEEDS:
            for algo in ("mcts", "mo_mcts"):
                for guard in (True, False):
                    specs.append(RunSpec(
                        family=FAMILY, algo=algo,
                        map_type=map_type, env_dim=env_dim, seed=seed,
                        overrides={"oscillation_guard": guard,
                                   "label": "on" if guard else "off"},
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
