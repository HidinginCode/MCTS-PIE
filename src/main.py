""" This module contains the main method that is used for non-cluster tests"""

#import matplotlib.pyplot as plt
#import networkx as nx
from state import State
from agent import Agent
from map import Map
from node import Node
from mc_tree import McTree
from controller import Controller

def main():
    """ Main method that runs all components togehter"""

    test_map = Map(map_dim=4, goal=(3,3))
    agent = Agent()
    controller = Controller(map_copy=test_map, current_agent=agent)

    root_state = State(state_controller=controller)
    root_node = Node(state=root_state)

    tree = McTree(root=root_node, max_depth=15)

    print("Starting MCTS ...")
    tree.run_search(200)
    #visualize_tree_hierarchical(root_node, max_depth=10, show_metrics=False, save_to_file=True)

# def hierarchy_pos(G, root=None, width=1.0, vert_gap=0.3, vert_loc=0, xcenter=0.5):
#     """
#     Position nodes in a hierarchy (tree layout).

#     Adapted from Joel's answer on StackOverflow:
#     https://stackoverflow.com/a/29597209/11699593
#     """
#     if not nx.is_tree(G):
#         raise TypeError("hierarchy_pos only works for trees")

#     if root is None:
#         if isinstance(G, nx.DiGraph):
#             root = next(iter(nx.topological_sort(G)))
#         else:
#             root = list(G.nodes)[0]

#     def _hierarchy_pos(graph, root, left, right, vert_loc, pos=None, parent=None):
#         if pos is None:
#             pos = {root: (xcenter, vert_loc)}
#         else:
#             pos[root] = ((left + right) / 2, vert_loc)
#         children = list(graph.successors(root))
#         if len(children) != 0:
#             dx = (right - left) / len(children)
#             nextx = left
#             for child in children:
#                 nextx += dx
#                 pos = _hierarchy_pos(graph, child, nextx - dx,
#                                      nextx, vert_loc - vert_gap, pos, root)
#         return pos

#     return _hierarchy_pos(graph, root, 0, width, vert_loc)


# def visualize_tree_hierarchical(root, max_depth=3, show_metrics=True, save_to_file=False):
#     """Visualize the MCTS tree in a hierarchical (top-down) layout."""
#     graph = nx.DiGraph()

#     def add_edges(node, depth=0):
#         if depth > max_depth:
#             return
#         node_id = id(node)
#         label = f"D{depth}\nV={node.visits}"
#         if show_metrics:
#             vals = ", ".join(f"{k}:{v:.2f}" for k, v in node.values.items())
#             label += f"\n{vals}\n|F|={len(node.front)}"
#         graph.add_node(node_id, label=label)
#         for child in (c for c in node.get_children().values() if c is not None):
#             graph.add_edge(node_id, id(child))
#             add_edges(child, depth + 1)

#     add_edges(root)
#     labels = nx.get_node_attributes(G, "label")
#     pos = hierarchy_pos(graph, root=id(root))  # <-- gives tree shape

#     plt.figure(figsize=(14, 8))
#     nx.draw(graph, pos, labels=labels, with_labels=True,
#             node_size=2500, node_color="lightblue",
#             font_size=7, arrows=True)
#     plt.title(f"MCTS Tree (max depth {max_depth})")

#     if save_to_file:
#         plt.savefig("mcts_tree.png", dpi=300)
#         print("Saved to mcts_tree.png")
#         plt.close()
#     else:
#         plt.show()

if __name__ == "__main__":
    main()
