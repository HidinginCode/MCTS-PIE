""" This module contains the main method that is used for non-cluster tests"""

#import networkx as nx
#from networkx.drawing.nx_pydot import graphviz_layout
#import matplotlib.pyplot as plt
from environment import Environment
from node import Node
from mc_tree import MctsTree
from controller import Controller
import moa_star
import os
import multiprocessing as mp
from astar import A_Star
import pickle

from analyzer import Analyzer

def main():
    """ Main method that runs all components togehter"""

    # 420 seed sweet spot 500 rollouts, 1000000, 20
    #random.seed(420)
    #np.random.seed(420)
    
    budgets = [10000, 100000, 1000000]
    per_sim_budgets = [10, 100, 1000]
    simulations_per_child = [100, 500, 1000]
    maps = ["easy_map", "checkerboard_map", "random_map"]

    for map in maps:
        for i in range(3):
            test_env = Environment(env_dim=10, goal=(9,9), map_type=map)
            controller = Controller(environment=test_env)
            root_node = Node(controller=controller)

            tree = MctsTree(root=root_node, max_solutions=20)

            print(f"Starting MCTS {i+1} on map {map} ...")

            tree.search(budgets[i], per_sim_budgets[i], simulations_per_child[i])
    # paths = moa_star.moa_star((0,0), (4,4), 5, moa_star.heuristic)
    # for path in paths: 
    #     path.reverse()
    #     print(path)
    #visualize_tree(tree._root, filename="mcts_tree.svg", max_depth=None)

def simulations(map: str, 
                env_dim, start: tuple, 
                goal: tuple, 
                budget: int, 
                per_sim_budget: int, 
                number_of_sims: int,
                rollout_method: int,
                root_selection_method: int, 
                tree_selection_method: int, 
                max_pareto_path_archive: int = 20,
                seed: int = 420) -> None:
    """Simulation method that is abled to set all hyperparameters.

    Args:
        map (str): Map name
        env_dim (_type_): Environment dimension (env_dim x env_dim)
        start (tuple): Start pos
        goal (tuple): Goal pos
        budget (int): Budget for number of simulations
        per_sim_budget (int): Budget per simulation of child node
        number_of_sims (int): Number of simulations per child node
        rollout_method (int): Indicator for rollout method
        root_selection_method (int): Root selection function (int is just indicator)
        tree_selection_method (int): Tree policy (int is just indicator)
        max_pareto_path_archive (int, optional): Maximum number of paths in pareto archive. Defaults to 20.
        seed (int, optional): Seed for pseudo randomness. Defaults to 420.
    """

    #random.seed(seed)
    #np.random.seed(seed)

    map_type = map


    env = Environment(env_dim = env_dim, goal = goal, map_type=map_type)
    controller = Controller(env, start_pos=start)
    root_node = Node(controller = controller)
    tree = MctsTree(root=root_node, seed = seed, max_solutions=max_pareto_path_archive)

    tree.search(total_budget=budget, per_sim_budget=per_sim_budget, simulations_per_child=number_of_sims, rollout_func = rollout_method, root_selection = root_selection_method, tree_selection = tree_selection_method)

def moa_wrapper(arguments: list):
    """Wrapper for multi-processing analysis of moa-star paths.

    Args:
        solutions (list): run arguments
    """
    print(f"Process: {os.getpid()}")
    if type(arguments) != list:
        arguments = [arguments]
    
    for args in arguments:
        #print(args, flush=True)
        map_type = args["map_type"]
        map_dim = args["map_dim"]
        start = args["start"]
        goal = args["goal"]

        solutions = (moa_star.moa_star(start=start, goal=goal, env_dim=map_dim, map_type=map_type))
        for x, solution in enumerate(solutions):
            if solution is not None:
                moves = []
                for i in range(len(solution)-1):
                    #print(solution[i+1][1])
                    moves.append(solution[i+1][1])
                env = Environment(env_dim=map_dim, goal=goal, map_type=map_type, start_pos=start)
                Analyzer.save_path_as_gif(environment=env, start_pos=start, moves=moves, gif_path=f"./solutions/solution-{map_type}-{map_dim}-{x}.gif")

def astar_wrapper(args: tuple) -> None:
    """Multiprocessing wrapper for A-Star

    Args:
        args (tuple): Arguments
    """
    map_type, dim = args
    print(f"Starting search for {map_type} {dim}x{dim}...")
    solver = A_Star(map_type, dim, (0,int(dim/2)), (dim-1,int(dim/2)))
    pareto = solver.epsilon_constraint_search()
    #solver.plot_pareto_front()


if __name__ == "__main__":
    #simulations(map = "easy_map", env_dim = 50, start = (0,25), goal=(49,25), budget=300000, per_sim_budget=75, number_of_sims=50, rollout_method=1, root_selection_method=0, tree_selection_method=0, seed=420)
    #run_experiments.sim_wrapper(("easy_map", 50, (0,25), (49,25), 300000, 75, 50, 1, 0, 0, 20, 420))
    # for dir in os.listdir("./log"):
    #    for file in os.listdir(f"./log/{dir}"):
    #        with open(f"./log/{dir}/{file}", "rb") as f:
    #            solutions = pickle.load(f)
    #            print(solutions)
    #            env = Environment(50, goal=(49,25), map_type="easy_map", start_pos=(0,25))
    #            moves = []
    #            for i in range(len(solutions[0]['moves'])):
    #                moves.append((solutions[0]['moves'][i], solutions[0]['shifts'][i]))
    #            print(moves)
    #            #Analyzer.interactive_step_path(env, start_pos=(0,25), moves=moves)
    #            Analyzer.save_path_as_gif(env, start_pos=(0,25), moves=moves, gif_path="./final_path.gif")
    # MAP_TYPES = ["meandering_river_map"]#["random_map", "easy_map", "checkerboard_map", "meandering_river_map"]
    # ENV_DIMS = [35]#, 50]

    # param_combinations = []
    # for map_type in MAP_TYPES:
    #     for dim in ENV_DIMS:
    #         param_combinations.append((map_type, dim))

    # with mp.Pool(1) as p:
    #     results = p.map(astar_wrapper, param_combinations)

    # analyzer = Analyzer()
    # os.makedirs("./map_graphics", exist_ok=True)
    # for file in os.listdir("./maps"):
    #     analyzer.visualize_map(f"./maps/{file}", f"./map_graphics/{file.removesuffix(".pickle")}.svg")
    #     print(f"Visualized {file}")

    # analyzer = Analyzer()
    # for dir in os.listdir("./log"):
    #     analyzer.extract_pareto_front(f"./log/{dir}")
    # analyzer.plot_pareto_maps()

    # for dir in os.listdir("./log"):
    #     for file in os.listdir(f"./log/{dir}"):
    #         filepath = f"./log/{dir}/{file}"
    #         with open(filepath, "rb") as f:
    #             data = pickle.load(f)
    #         if file.__contains__("ucb") and data==None:
    #             print("Ding")

    Analyzer.interactive_manual_control(
    map_type="checkerboard_map",   # or "easy_map", "checkerboard_map", "meandering_river_map"
    env_dim=51,
    start_pos=(0, 25),
    goal=(50, 25),           # optional, defaults to (env_dim-1, env_dim-1)
    )