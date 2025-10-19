""" This module contains the main method that is used for non-cluster tests"""

# import matplotlib.pyplot as plt
# import networkx as nx
from state import State
from agent import Agent
from map import Map
from node import Node
from mc_tree import McTree
from controller import Controller

def main():
    """ Main method that runs all components togehter"""

    test_map = Map(map_dim=5, goal=(4,4))
    agent = Agent()
    controller = Controller(map_copy=test_map, current_agent=agent)

    root_state = State(state_controller=controller)
    root_node = Node(state=root_state)

    tree = McTree(root=root_node)

    print("Starting MCTS ...")
    tree.run_search(20000)
#     visualize_tree_hierarchical(root_node, max_depth=10, show_metrics=False, save_to_file=True)


# def hierarchy_pos(graph, root=None, width=1.0, vert_gap=0.3, vert_loc=0, xcenter=0.5):
#     """Position nodes in a hierarchy (tree layout)."""
#     if not nx.is_tree(graph):
#         raise TypeError("hierarchy_pos only works for trees")

#     if root is None:
#         if isinstance(graph, nx.DiGraph):
#             root = next(iter(nx.topological_sort(graph)))
#         else:
#             root = list(graph.nodes)[0]

#     def _hierarchy_pos(graph, root, left, right, vert_loc, pos=None):
#         if pos is None:
#             pos = {root: (xcenter, vert_loc)}
#         else:
#             pos[root] = ((left + right) / 2, vert_loc)
#         children = list(graph.successors(root))
#         if children:
#             dx = (right - left) / len(children)
#             nextx = left
#             for child in children:
#                 nextx += dx
#                 pos = _hierarchy_pos(graph, child, nextx - dx,
#                                      nextx, vert_loc - vert_gap, pos)
#         return pos

#     return _hierarchy_pos(graph, root, 0, width, vert_loc)


# def visualize_tree_hierarchical(root, max_depth=3, show_metrics=True, save_to_file=False):
#     """Visualize the full MCTS tree in a hierarchical layout.
#     - Shows all nodes (up to max_depth)
#     - Displays visit counts and metrics
#     - Terminal nodes shown in red
#     """
#     graph = nx.DiGraph()
#     node_colors = {}

#     def add_edges(node, depth=0):
#         if depth > max_depth:
#             return
#         node_id = id(node)

#         # Label = Depth + Visit count
#         label = f"D{depth}\nV={node.get_visits()}"
#         if show_metrics:
#             vals = ", ".join(f"{k}:{v:.2f}" for k, v in node.get_values().items())
#             label += f"\n{vals}\n|F|={len(node.get_front())}"
#         graph.add_node(node_id, label=label)

#         # --- Color decision ---
#         if node.get_state().get_terminal_state():
#             node_colors[node_id] = "red"
#         else:
#             node_colors[node_id] = "lightblue"

#         # Add edges to children and recurse
#         for child in (c for c in node.get_children().values() if c is not None):
#             graph.add_edge(node_id, id(child))
#             add_edges(child, depth + 1)

#     # Build graph recursively
#     add_edges(root)

#     # Prepare layout + draw
#     labels = nx.get_node_attributes(graph, "label")
#     pos = hierarchy_pos(graph, root=id(root))

#     plt.figure(figsize=(14, 8))
#     nx.draw(
#         graph,
#         pos,
#         labels=labels,
#         with_labels=True,
#         node_size=2500,
#         node_color=[node_colors[n] for n in graph.nodes()],
#         font_size=7,
#         arrows=True,
#     )
#     plt.title(f"MCTS Tree (max depth {max_depth})")

#     if save_to_file:
#         plt.savefig("mcts_tree.png", dpi=300, bbox_inches="tight")
#         print("Saved to mcts_tree.png")
#         plt.close()
#     else:
#         plt.show()


if __name__ == "__main__":
    main()
