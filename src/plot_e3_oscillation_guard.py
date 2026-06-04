"""Plot E3 oscillation_guard ablation."""
from plot_e3_common import plot_axis

FAMILY_DIR = "./log/e3_ablations/oscillation_guard"
OUT_DIR = "./results/e3_ablations/oscillation_guard"


def main() -> None:
    plot_axis(FAMILY_DIR, OUT_DIR, "oscillation_guard")


if __name__ == "__main__":
    main()
