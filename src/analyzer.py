"""This module contains the analyzer class, used to log and create visual data."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Patch
import time
import os
import math
import networkx as nx
from controller import Controller
from environment import Environment
from PIL import Image
import tempfile
import pickle
import numpy as np
import re
from collections import defaultdict

class Analyzer:
    """Class that is used to log and create visual data."""
    
    def __init__(self):
        self._identifier = id(self)
        self.map_container = defaultdict(list)
    
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
    #  INTERACTIVE MANUAL CONTROL
    # ============================================================
    @staticmethod
    def interactive_manual_control(
        map_type: str,
        env_dim: int = 20,
        start_pos: tuple = (0, 0),
        goal: tuple = None,
    ):
        """Interactive visualization where the user manually moves the agent.

        Controls
        --------
        Arrow keys : movement direction (up=N, down=S, right=E, left=W)
        W/A/S/D    : shift direction (W=N, A=W, S=S, D=E)
        ENTER      : execute queued move + shift
        R          : reset to start
        ESC/Q      : close

        Args:
            map_type (str): Map type string, e.g. "random_map", "easy_map".
            env_dim (int): Grid dimension.
            start_pos (tuple): Agent start position.
            goal (tuple): Goal position; defaults to (env_dim-1, env_dim-1).
        """
        if goal is None:
            goal = (env_dim - 1, env_dim - 1)

        DIR_ARROWS = {
            (-1, 0): "↑",
            (1,  0): "↓",
            (0,  1): "→",
            (0, -1): "←",
        }
        KEY_TO_MOVE = {
            "up":    (-1, 0),
            "down":  (1,  0),
            "right": (0,  1),
            "left":  (0, -1),
        }
        KEY_TO_SHIFT = {
            "p": (-1, 0),   # shift north
            "ö": (1,  0),   # shift south
            "ä": (0,  1),   # shift east
            "l": (0, -1),   # shift west
        }

        env = Environment(env_dim=env_dim, goal=goal, new_env=True,
                          map_type=map_type, start_pos=start_pos)
        controller = Controller(environment=env, start_pos=start_pos)
        history = [tuple(start_pos)]

        fig, ax = plt.subplots(figsize=(7, 7))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        plt.subplots_adjust(right=0.72)

        env_data = controller._environment._environment
        nrows = len(env_data)
        ncols = len(env_data[0])

        im = ax.imshow(env_data, cmap="gray_r", aspect="equal",
                       vmin=0, vmax=1, interpolation="nearest")

        # Goal marker
        gr, gc = goal
        goal_patch = patches.Rectangle(
            (gc - 0.5, gr - 0.5), 1, 1,
            linewidth=2, edgecolor="#00ff88", facecolor="#00ff8840", zorder=3
        )
        ax.add_patch(goal_patch)
        ax.text(gc, gr, "G", ha="center", va="center",
                color="#00ff88", fontsize=9, fontweight="bold", zorder=4)

        # Start marker
        sr, sc = start_pos
        start_patch = patches.Rectangle(
            (sc - 0.5, sr - 0.5), 1, 1,
            linewidth=2, edgecolor="#ffaa00", facecolor="#ffaa0030", zorder=3
        )
        ax.add_patch(start_patch)
        ax.text(sc, sr, "S", ha="center", va="center",
                color="#ffaa00", fontsize=9, fontweight="bold", zorder=4)

        # Agent marker
        r0, c0 = start_pos
        agent_dot, = ax.plot([c0], [r0], "o", color="#ff4466",
                             markersize=10, markeredgecolor="white",
                             markeredgewidth=1.5, zorder=5)

        # Breadcrumb path line
        path_line, = ax.plot([], [], "-", color="#ff446680",
                             linewidth=1.5, zorder=4)

        ax.set_xticks(range(ncols))
        ax.set_yticks(range(nrows))
        ax.tick_params(labelsize=6, colors="#aaaaaa")
        ax.grid(color="#333355", linestyle="-", linewidth=0.4)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

        # Stats panel
        stats_ax = fig.add_axes([0.74, 0.30, 0.24, 0.60])
        stats_ax.set_facecolor("#0d0d1a")
        stats_ax.set_xticks([])
        stats_ax.set_yticks([])
        for spine in stats_ax.spines.values():
            spine.set_edgecolor("#334466")

        # Mutable state - defined before _make_stats_text
        state = {"move_dir": None, "shift_dir": None, "last_ok": None}

        def _make_stats_text():
            move_lbl  = DIR_ARROWS.get(state["move_dir"],  "—")
            shift_lbl = DIR_ARROWS.get(state["shift_dir"], "—")
            goal_flag = "✓ YES" if controller._goal_collected else "✗ NO"
            last_ok   = "✓ OK" if state["last_ok"] else ("✗ INVALID" if state["last_ok"] is False else "—")
            return (
                f"  OBJECTIVES\n"
                f"  {'─'*18}\n"
                f"  Steps        {controller.step_count:>6}\n"
                f"  Wt Shifted   {controller.weight_shifted:>6.2f}\n"
                f"  Dist→Goal    {controller.distance_to_goal:>6.2f}\n\n"
                f"  STATE\n"
                f"  {'─'*18}\n"
                f"  Position  ({controller.current_pos[0]:>2},{controller.current_pos[1]:>2})\n"
                f"  Goal      ({goal[0]:>2},{goal[1]:>2})\n"
                f"  Collected  {goal_flag}\n\n"
                f"  PENDING\n"
                f"  {'─'*18}\n"
                f"  Move dir    {move_lbl:>4}\n"
                f"  Shift dir   {shift_lbl:>4}\n"
                f"  Last move   {last_ok}\n\n"
                f"  CONTROLS\n"
                f"  {'─'*18}\n"
                f"  ←↑→↓  move dir\n"
                f"  IJKL   shift dir\n"
                f"  ENTER  execute\n"
                f"  R      reset\n"
                f"  ESC/Q  quit"
            )

        stats_txt = stats_ax.text(
            0.04, 0.97, _make_stats_text(),
            transform=stats_ax.transAxes,
            va="top", ha="left",
            fontsize=8.5, family="monospace",
            color="#cce0ff",
            linespacing=1.55,
        )

        def refresh():
            im.set_data(controller._environment._environment)
            r, c = controller.current_pos
            agent_dot.set_data([c], [r])
            rs = [p[0] for p in history]
            cs = [p[1] for p in history]
            path_line.set_data(cs, rs)
            stats_txt.set_text(_make_stats_text())
            fig.canvas.draw_idle()

        def on_key(event):
            nonlocal controller
            k = event.key

            if k in KEY_TO_MOVE:
                state["move_dir"] = KEY_TO_MOVE[k]

            elif k in KEY_TO_SHIFT:
                state["shift_dir"] = KEY_TO_SHIFT[k]

            elif k == "enter":
                if state["move_dir"] is None or state["shift_dir"] is None:
                    ax.set_title(
                        "Set BOTH move (arrow) and shift (WASD) first!",
                        color="#ff6666", fontsize=9
                    )
                    fig.canvas.draw_idle()
                    return
                ok = controller.move(state["move_dir"], state["shift_dir"])
                state["last_ok"] = ok
                if ok:
                    history.append(tuple(controller.current_pos))
                    state["move_dir"]  = None
                    state["shift_dir"] = None
                    title_color = "#aaffaa"
                    title_msg = (
                        f"Step {controller.step_count}  |  "
                        f"Pos {controller.current_pos}  |  "
                        f"Dist {controller.distance_to_goal:.1f}"
                    )
                    if controller._goal_collected:
                        title_msg = "GOAL REACHED!  " + title_msg
                        title_color = "#00ff88"
                else:
                    title_color = "#ff6666"
                    title_msg = "Invalid move — out of bounds"
                ax.set_title(title_msg, color=title_color, fontsize=9)

            elif k == "r":
                controller = Controller(environment=env, start_pos=start_pos)
                history.clear()
                history.append(tuple(start_pos))
                state.update(move_dir=None, shift_dir=None, last_ok=None)
                ax.set_title("Reset", color="#ffaa00", fontsize=9)

            elif k in ("escape", "q"):
                plt.close(fig)
                return

            refresh()

        fig.canvas.mpl_connect("key_press_event", on_key)
        ax.set_title(
            f"Map: {map_type}  |  {env_dim}x{env_dim}  |  Arrow=move  WASD=shift  ENTER=go",
            color="#aaaacc", fontsize=9
        )
        refresh()
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
        print(f"Steps: {controller.step_count}, Weight: {controller.weight_shifted}")

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

    @staticmethod
    def visualize_map(pickle_file: str, output_svg: str):
        # Load map
        with open(pickle_file, "rb") as f:
            env = pickle.load(f)

        filename = os.path.basename(pickle_file)

        # Remove extension
        name = os.path.splitext(filename)[0]

        # Remove dimension suffix
        name = re.sub(r"_\d+x\d+$", "", name)

        # Replace underscores with spaces
        name = name.replace("_", " ")

        print(name)

        env = np.array(env)
        env_dim = env.shape[0]

        # Start and goal positions
        start = (0, env_dim // 2)
        goal = (env_dim - 1, env_dim // 2)

        fig, ax = plt.subplots()

        # Plot environment
        im = ax.imshow(env, cmap="gray_r", origin="upper")

        # Draw start square
        start_square = Rectangle(
            (start[1] - 0.5, start[0] - 0.5),
            1, 1,
            facecolor="green",
            edgecolor="black",
            linewidth=1.5
        )
        ax.add_patch(start_square)

        # Draw goal square
        goal_square = Rectangle(
            (goal[1] - 0.5, goal[0] - 0.5),
            1, 1,
            facecolor="red",
            edgecolor="black",
            linewidth=1.5
        )
        ax.add_patch(goal_square)

        # Legend
        legend_elements = [
            Patch(facecolor="green", edgecolor="black", label="Start"),
            Patch(facecolor="red", edgecolor="black", label="Waypoint")
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        # Grid formatting
        ax.set_xticks(np.arange(-0.5, env_dim, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, env_dim, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.3)
        ax.tick_params(which="minor", bottom=False, left=False)

        ax.set_title(name.title())

        # Add colorbar
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Obstacle Weight")

        # Save SVG
        plt.savefig(output_svg, format="svg", bbox_inches="tight")
        plt.close()

    def plot_pareto_maps(self) -> None:
        """Plot pareto front for collected map dict."""
        graphics_path = f"./graphics/all_maps_mcts_wd/"
        os.makedirs(graphics_path, exist_ok=True)

        def pareto_filter(points):
            """Returns only non-dominated points (minimization for both objectives)."""
            pareto = []
            for p in points:
                dominated = False
                for q in points:
                    if (
                        q["values"]["step_count"] <= p["values"]["step_count"]
                        and q["values"]["weight_shifted"] <= p["values"]["weight_shifted"]
                    ) and (
                        q["values"]["step_count"] < p["values"]["step_count"]
                        or q["values"]["weight_shifted"] < p["values"]["weight_shifted"]
                    ):
                        dominated = True
                        break
                if not dominated:
                    pareto.append(p)
            return pareto

        # Collect all unique config labels across all maps first
        all_configs = set()
        for points in self.map_container.values():
            for p in points:
                label = (
                    f"tree={p['tree_selection_method']}\n"
                    f"root={p['root_selection_method']}\n"
                    f"sim={p['simulation_method']}"
                )
                all_configs.add(label)

        # Assign a unique, consistent color to each config
        color_map = {
            label: color
            for label, color in zip(
                sorted(all_configs),
                plt.cm.tab20.colors if len(all_configs) <= 20
                else plt.cm.hsv([i / len(all_configs) for i in range(len(all_configs))])
            )
        }

        for key, points in self.map_container.items():
            if not points:
                continue

            map_name, dim = key

            # Group points by their configuration
            config_groups = defaultdict(lambda: {"steps": [], "weights": []})
            for p in points:  # <- filter across ALL configs first
                label = (
                    f"tree={p['tree_selection_method']}\n"
                    f"root={p['root_selection_method']}\n"
                    f"sim={p['simulation_method']}"
                )
                config_groups[label]["steps"].append(p["values"]["step_count"])
                config_groups[label]["weights"].append(p["values"]["weight_shifted"])

            fig, ax = plt.subplots(figsize=(10, 6))

            for label, data in config_groups.items():
                ax.scatter(
                    data["steps"], data["weights"],
                    marker="o",
                    label=label,
                    color=color_map[label],
                )

            ax.set_xlabel("Step Count")
            ax.set_ylabel("Weight Shifted")
            ax.set_title(f"Pareto Front {map_name.replace("_", " ").title()} {dim}x{dim}")
            ax.grid(True)
            ax.legend(
                title="Configuration",
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                fontsize=7,
                title_fontsize=8,
            )

            plt.tight_layout()
            #plt.savefig(f"{graphics_path}/{map_name}-{dim}.svg", bbox_inches="tight")
            plt.close()
        
        ########## Print Output for Utility##########
        for (map_name, env_dim), points in self.map_container.items():
            if not points:
                continue

            filtered = pareto_filter(points)

            print(f"\n[{map_name} {env_dim}] total={len(points)} | pareto={len(filtered)}")

            config_groups = defaultdict(list)
            for p in filtered:
                config_key = f"{p['total_budget']}b/{p['per_sim_budget']}ps/{p['number_of_simulations']}n | tree={p['tree_selection_method']} | root={p['root_selection_method']} | sim={p['simulation_method']}"
                config_groups[config_key].append(p)

            for config_key, config_points in config_groups.items():
                print(f"  [{len(config_points):>3} pts]  {config_key}")
                # for p in config_points:
                #     steps = p["values"]["step_count"]
                #     weight = p["values"]["weight_shifted"]
                #     print(f"             steps={steps:<6} weight={weight:.4f}")
        input()

    def extract_pareto_front(self, dict_path: str) -> None:
        """Generates the Pareto front for the MCTS in form of immages.

        Args:
            dict_path (str): Path to the result dictionary
        """
        import matplotlib.pyplot as plt

        def plot_pareto_fronts(datapoints: dict) -> None:
            """
            Visualizes Pareto fronts stored in the datapoints dictionary.

            Args:
                datapoints (dict): {filename: [(step_count, weight_shifted), ...]}
            """

            for key, points in datapoints.items():
                if not points:
                    continue

                # Sort so the front is drawn nicely
                # points = sorted(points)
                steps = [p["values"]["step_count"] for p in points]
                weights = [p["values"]["weight_shifted"] for p in points]
                plt.figure()
                plt.scatter(steps, weights, marker="o", label=key)

                plt.xlabel("Step Count")
                plt.ylabel("Weight Shifted")
                plt.title("Pareto Front")
                plt.grid(True)

                #plt.savefig(f"{graphics_path}/{key}.svg")
                #print(f"{graphics_path}/{key}.svg")
                plt.close()

        def pareto_filter(points):
            """
            Filters a list of (step_count, weight_shifted) tuples and returns
            the Pareto optimal points (minimization for both objectives).
            """
            pareto = []

            for p in points:
                dominated = False
                for q in points:
                    if (q["values"]["step_count"] <= p["values"]["step_count"] and q["values"]["weight_shifted"] <= p["values"]["weight_shifted"]) and (q["values"]["step_count"] < p["values"]["step_count"] or q["values"]["weight_shifted"] < p["values"]["weight_shifted"]):
                        dominated = True
                        break

                if not dominated:
                    pareto.append(p)

            return pareto

        graphics_path = f"graphics/{dict_path.removeprefix("./")}"
        os.makedirs(graphics_path, exist_ok=True)
        datapoints = defaultdict(list)

        for file in os.listdir(dict_path):
            with open(f"{dict_path}/{file}", "rb") as f:
                new_filename = re.sub(r'-\d+\.pickle$', '', file)

                data = pickle.load(f)
                if data is not None:
                    for entry in data:
                        datapoints[new_filename].append((entry))
        
        for key in datapoints:
            for point in datapoints[key]:
                #print(point)
                self.map_container[(point["map_name"], point["env_dim"])].append(point)
        
        #print(self.map_container)

        for key in datapoints:
             datapoints[key] = pareto_filter(datapoints[key])
             #print(key)

        plot_pareto_fronts(datapoints=datapoints)