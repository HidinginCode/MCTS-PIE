#!/usr/bin/env python3
from mpi4py import MPI
import subprocess
import os

# -------------------------------------------------
# MPI initialization
# -------------------------------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()     # this worker's ID
size = comm.Get_size()     # total workers
root = 0

if rank == root:
    print(f"[MPI] Running with {size} workers")

# -------------------------------------------------
# Create output directory only once
# -------------------------------------------------
if rank == root:
    os.makedirs("mpi_logs", exist_ok=True)
comm.Barrier()


# -------------------------------------------------
# Build all parameter combinations
# -------------------------------------------------

PARAMS = []

MAP_TYPES = ["random_map"]
ENV_DIMS = [20, 35, 50]
TOTAL_BUDGETS = [1000000, 3000000, 5000000, 7000000, 9000000, 11000000]
PER_SIM_BUDGETS = [30, 50, 70, 90, 110, 130]
NUM_SIMS_LIST = [100, 300, 500, 700, 900, 1100]
TREE_METHODS = [0, 1, 2]
ROOT_METHODS = [0]
ROLLOUT_METHODS = [0, 1, 2]
SEEDS = [42, 420, 1337, 2024, 7777, 9999, 12345, 54321, 88888, 99999, 121212]
ARCHIVES = [20]

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
                                            "start_y": ENV // 2,
                                            "goal_x": ENV - 1,
                                            "goal_y": ENV // 2,
                                        })

total_jobs = len(PARAMS)

if rank == root:
    print(f"[MPI] Total parameter combinations: {total_jobs}")

comm.Barrier()


# -------------------------------------------------
# Worker loop: each rank processes jobs in stride
# -------------------------------------------------
for idx in range(rank, total_jobs, size):
    params = PARAMS[idx]

    # Construct log file name
    log_name = (
        f"mpi_logs/mcts_"
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

    # Execute the job
    with open(log_name, "w") as f:
        print("RUN:", " ".join(cmd), file=f)
        subprocess.run(cmd, stdout=f, stderr=f)

    print(f"[Rank {rank}] completed job {idx}/{total_jobs}")

comm.Barrier()

if rank == root:
    print("[MPI] All jobs finished.")
