"""Regenerate every figure / table from the produced CSVs."""

from __future__ import annotations

import argparse

import plot_e1_main
import plot_e2_anytime
import plot_e3_archive_cap
import plot_e3_oscillation_guard
import plot_e3_phase_split
import plot_e3_progressive_widening
import plot_e3_rollout
import plot_e3_root_selection
import plot_e3_tree_selection
import plot_e4_scaling

PLOTTERS = {
    "e1": [plot_e1_main],
    "e2": [plot_e2_anytime],
    "e3": [plot_e3_tree_selection, plot_e3_root_selection, plot_e3_rollout,
           plot_e3_progressive_widening, plot_e3_phase_split,
           plot_e3_archive_cap, plot_e3_oscillation_guard],
    "e4": [plot_e4_scaling],
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", type=str, default="")
    args = p.parse_args()
    keys = [k.strip() for k in args.only.split(",") if k.strip()] or list(PLOTTERS)
    for k in keys:
        for mod in PLOTTERS[k]:
            try:
                mod.main()
            except FileNotFoundError as e:
                print(f"skip {mod.__name__}: {e}")


if __name__ == "__main__":
    main()
