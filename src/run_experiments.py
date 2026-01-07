from main import simulations
import os
import multiprocessing as mp
from copy import deepcopy

def sim_wrapper(arg_list: tuple):
    print(f"Worker {os.getpid()} started working on {arg_list})")
    a,b,c,d,e,f,g,h,i,j,k,l = arg_list
    simulations(a,b,c,d,e,f,g,h,i,j,k,l)

def main():
    
    param_grid = generate_param_combinations()
    processes = 27
    print(f"Opening pool with {min(processes, os.cpu_count())} workers...")
    with mp.Pool(min(processes, os.cpu_count())) as p:
        results = p.map(sim_wrapper, list(param_grid))

    
def generate_param_combinations():
    MAP_TYPES = ["random_map"]
    ENV_DIMS = [35]
    TOTAL_BUDGETS = [100000, 200000, 300000, 400000, 500000, 600000]
    PER_SIM_BUDGETS = [100, 150, 200, 250, 300]
    NUM_SIMS_LIST = [100, 200, 300, 400]
    ROLLOUT_METHODS = [0]#, 1, 2]
    ROOT_METHODS = [0]
    TREE_METHODS = [0]
    ARCHIVES = [20]
    SEEDS = [42]

    print("Generating parameter grid ...")
    
    param_grid = []
    for m_type in MAP_TYPES:
        for dim in ENV_DIMS:
            start = (0, dim//2)
            goal = (dim-1, dim//2)
            for t_budget in TOTAL_BUDGETS:
                for s_budget in PER_SIM_BUDGETS:
                    for num_sim in NUM_SIMS_LIST:
                        for r_meth in ROLLOUT_METHODS:
                            for root_meth in ROOT_METHODS:
                                for tree_meth in TREE_METHODS:
                                    for arch in ARCHIVES:
                                        for seed in SEEDS:
                                            param_grid.append((m_type, dim, start, goal, t_budget, s_budget, num_sim, r_meth, root_meth, tree_meth, arch, seed))
    
    return param_grid

if __name__ == "__main__":
    main()
