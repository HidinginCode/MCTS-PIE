from main import simulations
import os
import multiprocessing as mp
from copy import deepcopy

def sim_wrapper(arg_list: tuple):
    print(f"Worker {os.getpid()} started working on {arg_list})")
    a,b,c,d,e,f,g,h,i,j,k,l = arg_list
    # We need to match the case arguments for function to the appropriate names
    # Rollout
    h_string = "None"

    # Child selection
    i_string = "None"

    # Root selection
    j_string = "None"
    print(f"h: {h}")
    print(f"i: {i}")
    print(f"j: {j}")

    match i:
        case 0: i_string = "hv_root_selection"
        case _: raise ValueError("Did not supply a suitable root selection indicator")

    match j:
        case 0: j_string = "ucb_child_selection"
        case 1: j_string = "pareto_path_child_selection_hv"
        case 2: j_string = "pareto_path_child_selection_cd"
        case 3: j_string = "pareto_path_child_selection_aec"
        case _: raise ValueError("Did not supply a suitable tree selection indicator")
    
    match h:
        case 0: h_string = "light_rollout"
        case 1: h_string = "iterative_heavy_square_sampling_rollout"
        case 2: h_string = "iterative_heavy_distance_weight_rollout"
        case _: raise ValueError("Did not supply a suitable rollout function indicator")
    
    out_path = f"./log/{a}-{b}-{h_string}/{e}-{f}-{g}-{j_string}-{i_string}-{h_string}-{l}.pickle"
    print(out_path)
    if not os.path.isfile(out_path):
        print(f"Starting simulation {a}-{b}-{h}/{e}-{f}-{g}-{i}-{j}-{h}-{l}", flush=True)
        simulations(a,b,c,d,e,f,g,h,i,j,k,l)
    else:
        print("Skipping...", flush=True)

def main():
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    processes = os.cpu_count()

    rank = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    world = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))

    param_grid = generate_param_combinations()
    my_grid = list(param_grid)[rank::world]

    print(f"[Shard {rank}/{world}] jobs={len(my_grid)} pool={cpus}")
    chunksize = max(10, len(my_grid) // (cpus * 8))

    with mp.Pool(max(2, cpus), maxtasksperchild=20) as p:
        p.map(sim_wrapper, my_grid, chunksize=chunksize)

    
def generate_param_combinations():
    MAP_TYPES = ["random_map", "easy_map", "checkerboard_map", "meandering_river_map"]
    ENV_DIMS = [35, 50]
    ROLLOUT_METHODS = [0, 1, 2]
    ROOT_METHODS = [0]
    TREE_METHODS = [0, 1, 2, 3]
    ARCHIVES = [20]
    SEEDS = [420, 1337, 9208, 2645, 1032, 7278, 5481, 9922, 7336, 2869, 1715, 7775, 721, 3624, 5599, 2644, 1080, 8460, 5652, 120, 6978, 5460, 4632, 1945, 3330, 6130, 989, 4368, 3742, 6852, 4138]

    # Three-level factor sets for Taguchi
    TOTAL_BUDGET_LEVELS   = [200000, 500000, 1000000]
    PER_SIM_BUDGET_LEVELS = [75, 100, 150]
    NUM_SIMS_LEVELS       = [25, 50, 100]

    # Taguchi L9 for (TOTAL_BUDGET, PER_SIM_BUDGET, NUM_SIMS)
    L9 = [
        (1, 1, 1),
        (1, 2, 2),
        (1, 3, 3),
        (2, 1, 2),
        (2, 2, 3),
        (2, 3, 1),
        (3, 1, 3),
        (3, 2, 1),
        (3, 3, 2),
    ]

    print("Generating Taguchi parameter grid ...")

    param_grid = []

    # Outer loop over Taguchi-designed triples
    for a, b, c in L9:
        total_budget   = TOTAL_BUDGET_LEVELS[a - 1]
        per_sim_budget = PER_SIM_BUDGET_LEVELS[b - 1]
        num_sims       = NUM_SIMS_LEVELS[c - 1]

        # Inner loops over ALL fixed scenario combinations
        for m_type in MAP_TYPES:
            for dim in ENV_DIMS:
                start = (0, dim // 2)
                goal = (dim - 1, dim // 2)
                for r_meth in ROLLOUT_METHODS:
                    for root_meth in ROOT_METHODS:
                        for tree_meth in TREE_METHODS:
                            for arch in ARCHIVES:
                                for seed in SEEDS:
                                    param_grid.append((
                                        m_type, dim, start, goal,
                                        total_budget, per_sim_budget, num_sims,
                                        r_meth, root_meth, tree_meth, arch, seed
                                    ))

    return param_grid


if __name__ == "__main__":
    main()
