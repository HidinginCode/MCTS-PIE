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

    random.seed(420)
    np.random.seed(420)

    print("Creating log directory ...")
    if os.path.exists("./log"):
        shutil.rmtree("./log")
    os.mkdir("./log")

    test_env = Environment(env_dim=4, goal=(3,3))
    controller = Controller(environment=test_env)
    root_node = Node(controller=controller)

    tree = MctsTree(root=root_node, max_solutions=10)

    print("Starting MCTS ...")

    tree.search(2000)
    # paths = moa_star.moa_star((0,0), (4,4), 5, moa_star.heuristic)
    # for path in paths: 
    #     path.reverse()
    #     print(path)
    #visualize_tree(tree._root, filename="mcts_tree.svg", max_depth=None)


if __name__ == "__main__":
    main()
