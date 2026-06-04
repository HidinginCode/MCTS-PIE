"""E3 ablation — rollout strategy (Light / SquareSampling / DistanceWeight)."""

from __future__ import annotations

import argparse

from experiments.config import MAP_SUITE, SEEDS, RunSpec
from experiments.runner import run

OUT_DIR = "./log/e3_ablations/rollout"
FAMILY = "e3_rollout"

# IDs: 0 Light, 1 SquareSampling, 2 DistanceWeight
ROLLOUT_VARIANTS = [
    ("mcts",    {"rollout": 0, "label": "light"}),
    ("mcts",    {"rollout": 1, "label": "square"}),
    ("mcts",    {"rollout": 2, "label": "distance_weight"}),
    ("mo_mcts", {"rollout": 0, "label": "light"}),
    ("mo_mcts", {"rollout": 1, "label": "square"}),
    ("mo_mcts", {"rollout": 2, "label": "distance_weight"}),
]


def build_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    for map_type, env_dim in MAP_SUITE:
        for seed in SEEDS:
            for algo, ov in ROLLOUT_VARIANTS:
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
