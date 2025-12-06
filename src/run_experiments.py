from main import simulations
import os
import multiprocessing as mp

def sim_wrapper(arg_list: tuple):
    print(f"Worker {os.getpid()} started working on {arg_list})")
    a,b,c,d,e,f,g,h,i,j,k,l = arg_list
    simulations(a,b,c,d,e,f,g,h,i,j,k,l)

def main():
    
    param_grid = generate_param_combinations()
    
    processes = 25
    print(f"Opening pool with {min(processes, os.cpu_count())} workers...")
    with mp.Pool(min(processes, os.cpu_count())) as p:
        results = p.map(sim_wrapper, list(param_grid))

    
def generate_param_combinations():
    MAP_TYPES = ["random_map"]
    ENV_DIMS = [20, 35, 50]
    TOTAL_BUDGETS = [1000000, 3000000, 5000000, 7000000, 9000000, 11000000]
    PER_SIM_BUDGETS = [30, 50, 70, 90, 110, 130]
    NUM_SIMS_LIST = [100, 300, 500, 700, 900, 1100]
    ROLLOUT_METHODS = [0]#, 1, 2]
    ROOT_METHODS = [0]
    TREE_METHODS = [0, 1, 2]
    ARCHIVES = [20]
    SEEDS = [42, 420, 1337, 2024, 7777, 9999, 12345, 54321, 88888, 99999, 121212]

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
