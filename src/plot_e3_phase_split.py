"""Plot E3 phase_split ablation."""
from plot_e3_common import plot_axis

FAMILY_DIR = "./log/e3_ablations/phase_split"
OUT_DIR = "./results/e3_ablations/phase_split"


def main() -> None:
    plot_axis(FAMILY_DIR, OUT_DIR, "phase_split")


if __name__ == "__main__":
    main()
