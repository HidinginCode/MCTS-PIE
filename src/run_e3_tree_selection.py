"""E3 ablation — tree selection strategy.

Sweeps UCB / HV / CD / AEGA on MCTS, plus GlobalHV on MO-MCTS.
"""

from __future__ import annotations

import argparse

from experiments.config import MAP_SUITE, SEEDS, RunSpec
from experiments.runner import run

OUT_DIR = "./log/e3_ablations/tree_selection"
FAMILY = "e3_tree_selection"

# Strategy IDs come from strategies/__init__.py:
#   0 UCB, 1 HV, 2 CD, 3 AEGA
TREE_SEL_VARIANTS = [
    ("mcts",    {"tree_selection": 0, "label": "ucb"}),
    ("mcts",    {"tree_selection": 1, "label": "hv"}),
    ("mcts",    {"tree_selection": 2, "label": "cd"}),
    ("mcts",    {"tree_selection": 3, "label": "aega"}),
    ("mo_mcts", {"tree_selection": 0, "label": "ucb"}),
    ("mo_mcts", {"tree_selection": 1, "label": "hv"}),
]


def build_specs() -> list[RunSpec]:
    specs: list[RunSpec] = []
    for map_type, env_dim in MAP_SUITE:
        for seed in SEEDS:
            for algo, ov in TREE_SEL_VARIANTS:
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
