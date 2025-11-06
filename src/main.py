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

def main():
    """ Main method that runs all components togehter"""

    random.seed(420)
    np.random.seed(420)

    test_env = Environment(env_dim=4, goal=(3,3))
    controller = Controller(environment=test_env)
    root_node = Node(controller=controller)

    tree = MctsTree(root=root_node)

    print("Starting MCTS ...")
    tree.search(4000)
    # paths = moa_star.moa_star((0,0), (4,4), 5, moa_star.heuristic)
    # for path in paths: 
    #     path.reverse()
    #     print(path)
    #visualize_tree(tree._root, filename="mcts_tree.svg", max_depth=None)



def visualize_tree(root, filename: str = "mcts_tree.svg", max_depth: int | None = None) -> None:
    """Visualize the MCTS tree using Graphviz (dot layout).

    Args:
        root (Node): Root node of the tree.
        filename (str): Output filename (.png, .pdf, .svg, etc.).
        max_depth (int | None): Optional cutoff depth for visualization.
    """

    G = nx.DiGraph()
    queue = [(root, 0)]

    # --- BFS traversal to build the graph ---
    while queue:
        node, depth = queue.pop(0)
        if max_depth is not None and depth > max_depth:
            continue

        node_id = node._identifier
        G.add_node(node_id)
        G.nodes[node_id]["label"] = f"Depth {node.get_depth()}\nVisits {node.get_visits()}"

        for child in node.get_children().values():
            if child is None:
                continue
            child_id = child.get_identificator()
            G.add_edge(node_id, child_id)
            G.nodes[child_id]["label"] = f"Depth {child.get_depth()}\nVisits {child.get_visits()}"
            queue.append((child, depth + 1))

    if len(G) == 0:
        print("[-] Tree is empty — nothing to visualize.")
        return

    # --- Compute hierarchical layout ---
    G.graph["graph"] = {
        "ranksep": "2.5",   # vertical spacing
        "nodesep": "1.5",   # horizontal spacing
    }
    pos = graphviz_layout(G, prog="dot")

    # --- Create large figure (no overlap) ---
    plt.figure(figsize=(500, 500))
    nx.draw(
        G,
        pos,
        with_labels=True,
        labels=nx.get_node_attributes(G, "label"),
        node_color="lightblue",
        node_size=3000,
        edgecolors="black",
        font_size=9,
        font_weight="bold",
    )
    plt.axis("off")
    plt.savefig(filename, format="svg")
    plt.close()

    print(f"[+] MCTS tree saved to {filename}")

def hierarchy_pos(graph, root=None, width=1.0, vert_gap=0.3, vert_loc=0, xcenter=0.5):
    """Position nodes in a hierarchy (tree layout)."""
    if not nx.is_tree(graph):
        raise TypeError("hierarchy_pos only works for trees")

    if root is None:
        if isinstance(graph, nx.DiGraph):
            root = next(iter(nx.topological_sort(graph)))
        else:
            root = list(graph.nodes)[0]

    def _hierarchy_pos(graph, root, left, right, vert_loc, pos=None):
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = ((left + right) / 2, vert_loc)
        children = list(graph.successors(root))
        if children:
            dx = (right - left) / len(children)
            nextx = left
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(graph, child, nextx - dx,
                                     nextx, vert_loc - vert_gap, pos)
        return pos

    return _hierarchy_pos(graph, root, 0, width, vert_loc)

def visualize_tree_hierarchical(root, max_depth=3, show_metrics=True, save_to_file=False):
    """Visualize the full MCTS tree in a hierarchical layout.
    - Shows all nodes (up to max_depth)
    - Displays visit counts and metrics
    - Terminal nodes shown in red
    """
    graph = nx.DiGraph()
    node_colors = {}

    def add_edges(node, depth=0):
        if depth > max_depth:
            return
        node_id = id(node)

        # Label = Depth + Visit count
        label = f"D{depth}\nV={node.get_visits()}"
        if show_metrics:
            vals = ", ".join(f"{k}:{v:.2f}" for k, v in node.get_values().items())
            label += f"\n{vals}\n|F|={len(node.get_front())}"
        graph.add_node(node_id, label=label)

        # --- Color decision ---
        if node.get_state().get_terminal_state():
            node_colors[node_id] = "red"
        else:
            node_colors[node_id] = "lightblue"

        # Add edges to children and recurse
        for child in (c for c in node.get_children().values() if c is not None):
            graph.add_edge(node_id, id(child))
            add_edges(child, depth + 1)

    # Build graph recursively
    add_edges(root)

    # Prepare layout + draw
    labels = nx.get_node_attributes(graph, "label")
    pos = hierarchy_pos(graph, root=id(root))

    plt.figure(figsize=(14, 8))
    nx.draw(
        graph,
        pos,
        labels=labels,
        with_labels=True,
        node_size=2500,
        node_color=[node_colors[n] for n in graph.nodes()],
        font_size=7,
        arrows=True,
    )
    plt.title(f"MCTS Tree (max depth {max_depth})")

    if save_to_file:
        plt.savefig("mcts_tree.png", dpi=300, bbox_inches="tight")
        print("Saved to mcts_tree.png")
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    main()
