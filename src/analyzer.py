"""This module contains the analyzer class, used to log and create visual data."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import os
import math
import networkx as nx
from controller import Controller
from environment import Environment
from PIL import Image
import tempfile
import pickle

class Analyzer:
    """Class that is used to log and create visual data."""
    
    def __init__(self):
        self._identifier = id(self)
    
    @property
    def identifier(self) -> int:
        return self._identifier
    
    # ============================================================
    #  HEATMAP VISUALIZATION (LIST-BASED ENVIRONMENT)
    # ============================================================
    @staticmethod
    def create_heatmap(environment: list, start: tuple, goal: tuple, path: list[tuple]):
        """Creates a heatmap of the environment and highlights start and goal."""
        print("Creating heatmap ...")

        data = environment
        nrows = len(data)
        ncols = len(data[0])

        fig, ax = plt.subplots()
        im = ax.imshow(data, aspect='equal', cmap="gray_r")
        plt.colorbar(im, ax=ax)

        # Highlight start & goal
        for (r, c) in [start, goal]:
            rect = patches.Rectangle(
                (c - 0.5, r - 0.5),
                1, 1,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)

        # Path dots
        for r, c in path:
            ax.plot(c, r, 'o', markersize=6, color='blue',
                    markeredgecolor='black', markeredgewidth=1.2)

        # Path arrows
        for (r1, c1), (r2, c2) in zip(path[:-1], path[1:]):
            ax.annotate(
                '',
                xy=(c1, r1),
                xytext=(c2, r2),
                arrowprops=dict(
                    arrowstyle='->',
                    linewidth=2.0,
                    color='white',
                    shrinkA=0, shrinkB=0
                )
            )

        # Grid and ticks
        ax.set_xticks(range(ncols))
        ax.set_yticks(range(nrows))
        ax.grid(color='black', linestyle='-', linewidth=0.5)

        plt.savefig(f"./log/{time.time()}.png")
        plt.close(fig)

    # ============================================================
    #  PATH WITH SHIFT VISUALIZATION
    # ============================================================
    @staticmethod
    def visualize_path_with_shifts(environment: list,
                                   path: list[tuple[int, int]],
                                   shift_dirs: list[tuple[int, int]],
                                   start: tuple[int, int],
                                   goal: tuple[int, int] | None = None,
                                   save_path: str | None = None):

        data = environment
        nrows = len(data)
        ncols = len(data[0])

        assert len(shift_dirs) == len(path) - 1

        fig, ax = plt.subplots()
        im = ax.imshow(data, cmap="gray_r", aspect="equal")
        plt.colorbar(im, ax=ax)

        # Highlight start + goal
        for pos in [start] + ([goal] if goal else []):
            if pos is None: 
                continue
            r, c = pos
            rect = patches.Rectangle((c-0.5, r-0.5), 1, 1,
                                    linewidth=2, edgecolor='red',
                                    facecolor='none')
            ax.add_patch(rect)

        # Path dots
        for r, c in path:
            ax.plot(c, r, 'o', color='blue', markersize=5,
                    markeredgecolor='black', markeredgewidth=1.0)

        # Movement arrows (white)
        for (r1, c1), (r2, c2) in zip(path[:-1], path[1:]):
            ax.arrow(c1, r1, c2 - c1, r2 - r1,
                     head_width=0.25, head_length=0.25,
                     fc='white', ec='black', linewidth=1.5,
                     length_includes_head=True, alpha=0.9)

        # Shift arrows (cyan)
        for (r, c), (dr, dc) in zip(path[1:], shift_dirs):
            r_target = r + dr
            c_target = c + dc

            if 0 <= r_target < nrows and 0 <= c_target < ncols:
                ax.arrow(c, r, dc * 0.8, dr * 0.8,
                         head_width=0.25, head_length=0.25,
                         fc='cyan', ec='black', linewidth=1.2,
                         linestyle='dashed', length_includes_head=True)
            else:
                ax.text(c, r, "×", color="red", ha="center", va="center", fontsize=8)

        # Grid
        ax.set_xticks(range(ncols))
        ax.set_yticks(range(nrows))
        ax.grid(color='black', linestyle='-', linewidth=0.5)

        if save_path is None:
            save_path = f"./log/heatmap_{time.time():.0f}.png"

        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)

    # ============================================================
    # INTERACTIVE STEPPER (list-based environment)
    # ============================================================
    @staticmethod
    def interactive_step_path(environment: Environment, start_pos: tuple[int, int], moves: list):
        controller = Controller(environment, start_pos)

        fig, ax = plt.subplots()
        im = ax.imshow(controller._environment._environment, cmap="gray_r", aspect="equal")

        env_data = controller._environment._environment
        nrows = len(env_data)
        ncols = len(env_data[0])

        ax.set_xticks(range(ncols))
        ax.set_yticks(range(nrows))
        ax.grid(color='black', linestyle='-', linewidth=0.5)

        r0, c0 = controller._current_pos
        agent_marker, = ax.plot([c0], [r0], 'o', color='red', markersize=8,
                                markeredgecolor='black', markeredgewidth=1.2)

        state = {"i": 0}

        def update_plot():
            im.set_data(controller._environment._environment)
            r, c = controller._current_pos
            agent_marker.set_data([c], [r])
            ax.set_title(f"Step {state['i']}/{len(moves)}")
            fig.canvas.draw_idle()

        def on_key(event):
            if event.key == "right":
                if state["i"] < len(moves):
                    move_dir, shift_dir = moves[state["i"]]
                    controller.move(move_dir, shift_dir)
                    update_plot()
                    state["i"] += 1
            elif event.key == "escape":
                plt.close(fig)

        fig.canvas.mpl_connect("key_press_event", on_key)
        update_plot()
        plt.show()

    # ============================================================
    #  MCTS TREE VISUALIZATION (unchanged, list-safe)
    # ============================================================
    @staticmethod
    def visualize_mcts_svg(root, filename="mcts_tree.svg",
                           max_depth=None, show_metrics=True):
        """Same functionality, no NumPy usage anywhere."""
        print("Visualizing tree...")

        def hierarchy_pos(graph, root_id, width=1.0, vert_gap=0.5, vert_loc=0.0):
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

        G = nx.DiGraph()
        visited = set()
        queue = [(root, 0)]

        while queue:
            node, depth = queue.pop(0)
            node_id = id(node)

            if node_id in visited:
                continue
            visited.add(node_id)

            if max_depth is not None and depth > max_depth:
                continue

            # Label
            label = f"D={getattr(node, '_depth', '?')}\\nV={getattr(node, '_visits', 0)}"
            if show_metrics:
                vals = getattr(node, '_values', {})
                if isinstance(vals, dict):
                    label += "\\n" + ", ".join(f"{k}:{float(v):.2f}" for k, v in vals.items())

            # Terminal?
            try:
                is_term = bool(node.is_terminal_state())
            except:
                is_term = False

            G.add_node(node_id, label=label, terminal=is_term)

            # Edges
            children = getattr(node, '_children', {})
            for mv, child in children.items():
                if child is None:
                    continue
                cid = id(child)
                try:
                    m1, m2 = mv
                    edge_label = f"{getattr(m1, 'name', m1)}|{getattr(m2, 'name', m2)}"
                except:
                    edge_label = ""
                G.add_edge(node_id, cid, elabel=edge_label)
                queue.append((child, depth + 1))

        if G.number_of_nodes() == 0:
            return

        try:
            from networkx.drawing.nx_pydot import graphviz_layout
            pos = graphviz_layout(G, prog="dot")
        except:
            pos = hierarchy_pos(G, id(root))

        base = 10
        scale = 1.0 + math.log2(max(G.number_of_nodes(), 2))
        plt.figure(figsize=(base * 0.5 * scale, base * 0.3 * scale))

        node_labels = nx.get_node_attributes(G, 'label')
        node_colors = ["#f28e2b" if G.nodes[n]["terminal"] else "#4e79a7" for n in G.nodes()]

        nx.draw(
            G, pos, labels=node_labels,
            node_color=node_colors,
            node_size=2000,
            edgecolors="black",
            font_size=8,
            font_weight="bold",
            arrows=True,
            width=1.2
        )

        edge_labels = nx.get_edge_attributes(G, 'elabel')
        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        plt.axis("off")
        plt.savefig(filename, format="svg", bbox_inches="tight")
        plt.close()

    # ============================================================
    #  SAVE GIF (list-based environment)
    # ============================================================
    @staticmethod
    def save_path_as_gif(environment: Environment,
                         start_pos: tuple[int, int],
                         moves: list,
                         gif_path="path.gif",
                         frame_duration=300):

        controller = Controller(environment, start_pos)

        fig, ax = plt.subplots()
        im = ax.imshow(controller._environment._environment, cmap="gray_r", aspect="equal")

        env_data = controller._environment._environment
        nrows = len(env_data)
        ncols = len(env_data[0])

        ax.set_xticks(range(ncols))
        ax.set_yticks(range(nrows))
        ax.grid(color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()

        r0, c0 = controller._current_pos
        agent_marker, = ax.plot([c0], [r0], 'o', color='red', markersize=8,
                                markeredgecolor='black', markeredgewidth=1.2)

        frames = []
        tmpdir = tempfile.mkdtemp()

        def save_frame(step: int):
            ax.set_title(f"Step {step}/{len(moves)}")
            fig.canvas.draw()
            frame_path = os.path.join(tmpdir, f"frame_{step:05d}.png")
            fig.savefig(frame_path, dpi=120)
            frames.append(frame_path)

        save_frame(0)

        for i, (move_dir, shift_dir) in enumerate(moves, start=1):
            controller.move(move_dir, shift_dir)

            im.set_data(controller._environment._environment)
            r, c = controller._current_pos
            agent_marker.set_data([c], [r])

            save_frame(i)

        plt.close(fig)

        # Assemble GIF
        images = [Image.open(f) for f in frames]
        images[0].save(
            gif_path, save_all=True, append_images=images[1:],
            duration=frame_duration, loop=0
        )

        for f in frames:
            os.remove(f)

        return gif_path
    
    @staticmethod
    def visualize_maps() -> None:
        """Method that creates a heatmap of all maps in ./maps
        """

        out_path = "./out"
        map_path = "./maps"

        if not os.path.exists(out_path):
            os.mkdir(out_path)

        if not os.path.exists(map_path):
            raise FileNotFoundError("Directory ./maps does not seem to exist.")
        
        for dir in os.listdir(map_path):
            path = os.path.join(map_path, dir)
            with open(path, "rb") as f:
                map_name = dir.removesuffix(".pickle").replace("_", " ")
                map_array = pickle.load(f)
                plt.imshow(map_array, cmap="gray_r", interpolation="nearest")
                plt.colorbar()
                plt.title(f"{map_name}")
                plt.savefig(os.path.join(out_path, dir.removesuffix(".pickle")+".png"))
                plt.close()