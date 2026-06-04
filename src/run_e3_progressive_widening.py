"""E3 ablation — progressive widening (C, alpha)."""

from __future__ import annotations

import argparse

from experiments.config import MAP_SUITE, SEEDS, RunSpec
from experiments.runner import run

OUT_DIR = "./log/e3_ablations/progressive_widening"
FAMILY = "e3_progressive_widening"

PW_VARIANTS = [
    {"pw_c": 1.0, "pw_alpha": 0.5, "label": "C1.0_a0.5"},
    {"pw_c": 1.5, "pw_alpha": 0.5, "label": "C1.5_a0.5"},
    {"pw_c": 2.0, "pw_alpha": 0.5, "label": "C2.0_a0.5"},
    {"pw_c": 1.5, "pw_alpha": 0.3, "label": "C1.5_a0.3"},
    {"pw_c": 1.5, "pw_alpha": 0.7, "label": "C1.5_a0.7"},
]


def build_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    for map_type, env_dim in MAP_SUITE:
        for seed in SEEDS:
            for algo in ("mcts", "mo_mcts"):
                for ov in PW_VARIANTS:
                    specs.append(RunSpec(
                        family=FAMILY, algo=algo,
                        map_type=map_type, env_dim=env_dim, seed=seed,
                        overrides=dict(ov),
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
