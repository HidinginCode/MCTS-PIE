"""Plot E3 archive_cap ablation."""
from plot_e3_common import plot_axis

FAMILY_DIR = "./log/e3_ablations/archive_cap"
OUT_DIR = "./results/e3_ablations/archive_cap"


def main() -> None:
    plot_axis(FAMILY_DIR, OUT_DIR, "archive_cap")


if __name__ == "__main__":
    main()
