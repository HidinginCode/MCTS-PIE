import itertools
import csv

MAP_TYPES = ["random_map"]
ENV_DIMS = [20, 35, 50]
TOTAL_BUDGETS = [3000000, 5000000, 7000000, 9000000, 11000000]
PER_SIM_BUDGETS = [30, 50, 70, 90, 110, 130]
NUM_SIMS_LIST = [100, 300, 500, 700, 900, 1100]
TREE_METHODS = [0, 1, 2]
ROOT_METHODS = [0]
ROLLOUT_METHODS = [0, 1, 2]
SEEDS = [42, 420, 1337, 2024, 7777, 9999, 12345, 54321, 88888, 99999, 121212]
ARCHIVES = [20]

param_grid = itertools.product(
    MAP_TYPES,
    ENV_DIMS,
    TOTAL_BUDGETS,
    PER_SIM_BUDGETS,
    NUM_SIMS_LIST,
    TREE_METHODS,
    ROOT_METHODS,
    ROLLOUT_METHODS,
    SEEDS,
    ARCHIVES
)

with open("params.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "map_type", "env_dim", "budget", "per_sim_budget",
        "num_sims", "tree_method", "root_method",
        "rollout_method", "seed", "archive"
    ])
    for row in param_grid:
        writer.writerow(row)

print("Generated params.csv")
