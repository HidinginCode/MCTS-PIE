"""Plot E3 tree-selection ablation."""
from plot_e3_common import plot_axis

FAMILY_DIR = "./log/e3_ablations/tree_selection"
OUT_DIR = "./results/e3_ablations/tree_selection"


def main() -> None:
    plot_axis(FAMILY_DIR, OUT_DIR, "tree_selection")


if __name__ == "__main__":
    main()
