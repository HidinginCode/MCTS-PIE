"""E3 ablation — root selection (HVRootSelection vs GlobalHVRootSelection)."""

from __future__ import annotations

import argparse

from experiments.config import MAP_SUITE, SEEDS, RunSpec
from experiments.runner import run

OUT_DIR = "./log/e3_ablations/root_selection"
FAMILY = "e3_root_selection"

ROOT_VARIANTS = [
    ("mo_mcts", {"root_selection": 0, "label": "hv"}),
    ("mo_mcts", {"root_selection": 1, "label": "global_hv"}),
]


def build_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    for map_type, env_dim in MAP_SUITE:
        for seed in SEEDS:
            for algo, ov in ROOT_VARIANTS:
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
