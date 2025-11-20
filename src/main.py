""" This module contains the main method that is used for non-cluster tests"""

import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt
from environment import Environment
from node import Node
from mc_tree import MctsTree
from controller import Controller
import random
import numpy as np
import moa_star
import os
import shutil

def main():
    """ Main method that runs all components togehter"""

    # 420 seed sweet spot 500 rollouts, 1000000, 20
    #random.seed(420)
    #np.random.seed(420)
    
    budgets = [10000, 100000, 1000000]
    per_sim_budgets = [10, 100, 1000]
    simulations_per_child = [100, 500, 1000]


    for i in range(3):
        test_env = Environment(env_dim=10, goal=(9,9), map_type="easy_map")
        print(test_env._map_type)
        controller = Controller(environment=test_env)
        root_node = Node(controller=controller)
        print(root_node._controller._environment._map_type)

        tree = MctsTree(root=root_node, max_solutions=20)

        print(f"Starting MCTS {i} ...")

        tree.search(budgets[i], per_sim_budgets[i], simulations_per_child[i])
    # paths = moa_star.moa_star((0,0), (4,4), 5, moa_star.heuristic)
    # for path in paths: 
    #     path.reverse()
    #     print(path)
    #visualize_tree(tree._root, filename="mcts_tree.svg", max_depth=None)


if __name__ == "__main__":
    main()
