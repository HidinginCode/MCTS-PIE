""" This module contains the main method that is used for non-cluster tests"""

from state import State
from agent import Agent
from map import Map
from node import Node
from mc_tree import McTree
from controller import Controller

def main():
    """ Main method that runs all components togehter"""

    test_map = Map(map_dim=2, goal=(1,1))
    agent = Agent()
    controller = Controller(map_copy=test_map, current_agent=agent)

    root_state = State(state_controller=controller)
    root_node = Node(state=root_state)

    tree = McTree(root=root_node, max_depth=15)

    print("Starting MCTS ...")
    tree.run_search(1000000)

if __name__ == "__main__":
    main()
