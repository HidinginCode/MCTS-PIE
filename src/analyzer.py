"""This module contains the analyzer class, used to log and create visual data."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import os
import math
import networkx as nx

class Analyzer():
    """Class that is used to log and create visual data."""
    
    def __init__(self):
        self._identifier = id(self)
    
    @property
    def identifier(self) -> int:
        """Getter for identifier property.

        Returns:
            int: ID
        """
        return self._identifier
    
    @staticmethod
    def create_heatmap(environment: np.ndarray, start: tuple, goal: tuple, path: list[tuple]):
        """Creates a heatmap of the environment and highlights start and goal.

        Args:
            environment (np.ndarray): Environment after last step.
            start (tuple): Tuple of start coordinates.
            goal (tuple): Tuple of goal coordinates.
        """
        print("Creating heatmap ...")
        data = environment
        highlighted_cells = [start, goal]
        fig, ax = plt.subplots()
        im = ax.imshow(data, aspect='equal',cmap="gray_r")


        plt.colorbar(im, ax=ax)

        # Highlight the chosen squares
        for (r, c) in highlighted_cells:
            rect = patches.Rectangle(
                (c - 0.5, r - 0.5),  # x, y position of the lower-left corner
                1, 1,                # width, height
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)

        for r, c in path:
            ax.plot(c, r, 'o', markersize=6, color='blue', markeredgecolor='black', markeredgewidth=1.2)  # Dot in the cell


        for (r1, c1), (r2, c2) in zip(path[:-1], path[1:]):
            ax.annotate(
                '', 
                xy=(c1, r1), 
                xytext=(c2, r2),
                arrowprops=dict(arrowstyle='->', 
                                linewidth=2.0,              
                                color='white',              
                                shrinkA=0, shrinkB=0)
            )


        ax.set_xticks(np.arange(-0.5, data.shape[1], 1))
        ax.set_yticks(np.arange(-0.5, data.shape[0], 1))
        ax.grid(color='black', linestyle='-', linewidth=0.5)
        # Optional: Add title and better ticks
        ax.set_title("Heatmap of Environment")
        ax.set_xticks(range(data.shape[1]))
        ax.set_yticks(range(data.shape[0]))

        plt.savefig(f"./log/{time.time()}.png")


    @staticmethod
    def visualize_mcts_svg(
        root,
        filename: str = "mcts_tree.svg",
        max_depth: int | None = None,
        show_metrics: bool = True
    ) -> None:
        """
        Render the current MCTS tree (rooted at `root`) to an SVG file.

        This uses NetworkX for graph construction and attempts to use Graphviz's
        'dot' layout for a clean top-down tree view. If Graphviz isn't installed,
        it falls back to a simple hierarchical layout calculated in Python.

        Args:
            root (Node): Root node of the MCTS tree.
            filename (str): Output path to the SVG file.
            max_depth (int | None): Limit visualization to a certain depth.
            show_metrics (bool): If True, show visits & values in node labels.
        """
        print("Visualizing tree...")

        def hierarchy_pos(graph, root_id, width=1.0, vert_gap=0.5, vert_loc=0.0):
            """Basic top-down hierarchical layout if Graphviz is not available."""
            def _hierarchy(n, left, right, y, pos):
                children = list(graph.successors(n))
                pos[n] = ((left + right) / 2.0, y)
                if children:
                    dx = (right - left) / max(1, len(children))
                    new_left = left
                    for c in children:
                        c_right = new_left + dx
                        _hierarchy(c, new_left, c_right, y - vert_gap, pos)
                        new_left = c_right
                return pos
            return _hierarchy(root_id, 0.0, width, vert_loc, {})

        # Build directed graph from nodes
        G = nx.DiGraph()
        visited = set()
        queue = [(root, 0)]

        while queue:
            node, depth = queue.pop(0)

            if id(node) in visited:
                continue
            visited.add(id(node))

            if max_depth is not None and depth > max_depth:
                continue

            # Build node label
            node_id = id(node)
            label = f"D={getattr(node, '_depth', '?')}\\nV={getattr(node, '_visits', 0)}"
            if show_metrics:
                values = getattr(node, '_values', {})
                if isinstance(values, dict) and values:
                    val_str = ", ".join(f"{k}:{float(v):.2f}" for k, v in values.items())
                    label += f"\\n{val_str}"

            # Mark terminal vs non-terminal
            is_term = False
            try:
                is_term = bool(node.is_terminal_state())
            except Exception:
                pass

            G.add_node(node_id, label=label, terminal=is_term)

            # Add edges
            children = getattr(node, '_children', {})
            if isinstance(children, dict):
                for move, child in children.items():
                    if child is None:
                        continue
                    child_id = id(child)
                    try:
                        m1, m2 = move
                        edge_label = f"{getattr(m1, 'name', m1)}|{getattr(m2, 'name', m2)}"
                    except Exception:
                        edge_label = ""
                    G.add_edge(node_id, child_id, elabel=edge_label)
                    queue.append((child, depth + 1))

        if G.number_of_nodes() == 0:
            return

        # Try Graphviz layout, else fallback
        try:
            from networkx.drawing.nx_pydot import graphviz_layout
            pos = graphviz_layout(G, prog="dot")
        except Exception:
            pos = hierarchy_pos(G, id(root))

        # Figure size scales with number of nodes (just for readability)
        base = 10
        scale = 1.0 + math.log2(max(G.number_of_nodes(), 2))
        plt.figure(figsize=(base * 0.5 * scale, base * 0.3 * scale))

        node_labels = nx.get_node_attributes(G, 'label')
        node_colors = [
            "#f28e2b" if G.nodes[n].get("terminal") else "#4e79a7"
            for n in G.nodes()
        ]

        nx.draw(
            G,
            pos,
            labels=node_labels,
            node_color=node_colors,
            node_size=2000,
            edgecolors="black",
            font_size=8,
            font_weight="bold",
            arrows=True,
            width=1.2
        )

        # Draw edge labels if present
        edge_labels = nx.get_edge_attributes(G, 'elabel')
        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        plt.axis("off")
        plt.savefig(filename, format="svg", bbox_inches="tight")
        plt.close()