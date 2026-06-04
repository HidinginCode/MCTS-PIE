"""Plot E3 progressive_widening ablation."""
from plot_e3_common import plot_axis

FAMILY_DIR = "./log/e3_ablations/progressive_widening"
OUT_DIR = "./results/e3_ablations/progressive_widening"


def main() -> None:
    plot_axis(FAMILY_DIR, OUT_DIR, "progressive_widening")


if __name__ == "__main__":
    main()
