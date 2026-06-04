"""Plot E3 rollout ablation."""
from plot_e3_common import plot_axis

FAMILY_DIR = "./log/e3_ablations/rollout"
OUT_DIR = "./results/e3_ablations/rollout"


def main() -> None:
    plot_axis(FAMILY_DIR, OUT_DIR, "rollout")


if __name__ == "__main__":
    main()
