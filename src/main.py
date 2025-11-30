""" This module contains the main method that is used for non-cluster tests"""

#import networkx as nx
#from networkx.drawing.nx_pydot import graphviz_layout
#import matplotlib.pyplot as plt
from environment import Environment
from node import Node
from mc_tree import MctsTree
from controller import Controller
import random
import numpy as np
import moa_star
import os
import shutil
#from analyzer import Analyzer

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

    random.seed(seed)
    np.random.seed(seed)

    map_type = map


    env = Environment(env_dim = env_dim, goal = goal, map_type=map_type)
    controller = Controller(env, start_pos=start)
    root_node = Node(controller = controller)
    tree = MctsTree(root=root_node, seed = seed, max_solutions=max_pareto_path_archive)

    tree.search(total_budget=budget, per_sim_budget=per_sim_budget, simulations_per_child=number_of_sims, rollout_func = rollout_method, root_selection = root_selection_method, tree_selection = tree_selection_method)

if __name__ == "__main__":
    simulations(map = "easy_map", env_dim = 20, start = (0,0), goal=(9,9), budget=1000000, per_sim_budget=20, number_of_sims=1000, rollout_method=0, root_selection_method=0, tree_selection_method=0)
    #Analyzer.visualize_maps()