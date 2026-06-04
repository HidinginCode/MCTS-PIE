"""Single entry point that runs every experiment family in order.

Usage::

    python run_all_experiments.py --workers 8
    python run_all_experiments.py --only e1,e3
    python run_all_experiments.py --no-resume
"""

from __future__ import annotations

import argparse

import run_e1_main_comparison
import run_e2_anytime
import run_e3_ablations
import run_e4_scaling

FAMILIES = {
    "e1": run_e1_main_comparison,
    "e2": run_e2_anytime,
    "e3": run_e3_ablations,
    "e4": run_e4_scaling,
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--only", type=str, default="",
                   help="comma-separated family keys (e.g. 'e1,e3')")
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()

    keys = [k.strip() for k in args.only.split(",") if k.strip()] or list(FAMILIES)
    for k in keys:
        if k not in FAMILIES:
            raise SystemExit(f"unknown family '{k}'; choices: {list(FAMILIES)}")
        print(f"\n#### {k} ####")
        FAMILIES[k].main(n_workers=args.workers, resume=not args.no_resume)


if __name__ == "__main__":
    main()
