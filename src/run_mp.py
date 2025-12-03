import multiprocessing as mp
import subprocess
import argparse
import os

PARAMS = []

MAP_TYPES = ["random_map"]
ENV_DIMS = [20, 30, 50]
TOTAL_BUDGETS = [1000000, 5000000, 10000000]
PER_SIM_BUDGETS = [30, 50, 100]
NUM_SIMS_LIST = [100, 250, 500]
TREE_METHODS = [0, 1, 2]
ROOT_METHODS = [0]
ROLLOUT_METHODS = [0, 1, 2]
SEEDS = [42, 420, 1337, 2024, 7777, 9999, 12345, 54321, 88888, 99999]
ARCHIVES = [20]

# Build combination list
for MAP in MAP_TYPES:
    for ENV in ENV_DIMS:
        for BUDGET in TOTAL_BUDGETS:
            for PER_SIM in PER_SIM_BUDGETS:
                for NUM_SIMS in NUM_SIMS_LIST:
                    for TREESEL in TREE_METHODS:
                        for ROOTSEL in ROOT_METHODS:
                            for ROLLOUT in ROLLOUT_METHODS:
                                for SEED in SEEDS:
                                    for ARCHIVE in ARCHIVES:
                                        PARAMS.append({
                                            "map": MAP,
                                            "env_dim": ENV,
                                            "budget": BUDGET,
                                            "per_sim": PER_SIM,
                                            "num_sims": NUM_SIMS,
                                            "tree_sel": TREESEL,
                                            "root_sel": ROOTSEL,
                                            "rollout": ROLLOUT,
                                            "seed": SEED,
                                            "archive": ARCHIVE,
                                            "start_x": 0,
                                            "start_y": ENV//2,
                                            "goal_x": ENV-1,
                                            "goal_y": ENV//2,
                                        })

print(f"Loaded {len(PARAMS)} parameter combinations.")

def estimate_memory_per_worker_mb():
    return 8000

def compute_safe_worker_count(max_processes, total_memory_mb):
    mem_per_worker = estimate_memory_per_worker_mb()
    max_by_mem = total_memory_mb // mem_per_worker
    return max(1, min(max_processes, max_by_mem))


def run_case(params):
    log_name = (
        f"mp_logs/mcts_"
        f"{params['map']}_"
        f"{params['env_dim']}_"
        f"{params['budget']}_"
        f"{params['per_sim']}_"
        f"{params['num_sims']}_"
        f"{params['tree_sel']}_"
        f"{params['root_sel']}_"
        f"{params['rollout']}_"
        f"{params['seed']}.log"
    )


    cmd = [
        "python3", "main_cluster.py",
        "--map", params["map"],
        "--env_dim", str(params["env_dim"]),
        "--start_x", str(params["start_x"]),
        "--start_y", str(params["start_y"]),
        "--goal_x", str(params["goal_x"]),
        "--goal_y", str(params["goal_y"]),
        "--budget", str(params["budget"]),
        "--per_sim_budget", str(params["per_sim"]),
        "--num_sims", str(params["num_sims"]),
        "--rollout_method", str(params["rollout"]),
        "--root_sel", str(params["root_sel"]),
        "--tree_sel", str(params["tree_sel"]),
        "--max_archive", str(params["archive"]),
        "--seed", str(params["seed"]),
    ]

    with open(log_name, "w") as f:
        print("RUN:", " ".join(cmd), file=f)
        subprocess.run(cmd, stdout=f, stderr=f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int)
    parser.add_argument("--total_memory")
    args = parser.parse_args()

    total_memory_mb = int(args.total_memory.rstrip("G")) * 1024
    processes = 24

    print(f"Using {24} workers (memory-safe).")

    os.makedirs("mp_logs", exist_ok=True)

    with mp.Pool(24) as pool:
        pool.map(run_case, PARAMS)
