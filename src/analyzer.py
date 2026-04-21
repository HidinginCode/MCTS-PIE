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

ANALYZER_RCPARAMS = {
    "font.size":       9,
    "axes.titlesize":  10,
    "axes.labelsize":  9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
}

PARETO_PLOT_RCPARAMS = {
    "font.size":             10,
    "axes.titlesize":        11,
    "axes.labelsize":        10,
    "xtick.labelsize":       9,
    "ytick.labelsize":       9,
    "legend.fontsize":       7,
    "legend.title_fontsize": 7,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
}

TREE_METHOD_ABBREVIATIONS = {
    "ucb_child_selection":             "UCB",
    "pareto_path_child_selection_aec": "AEC",
    "pareto_path_child_selection_cd":  "CD",
    "pareto_path_child_selection_hv":  "HV",
}

# Substrings to strip from simulation method names, in order
_SIM_STRIP = [
    "iterative_heavy_",
    "_rollout",
    "rollout",
]

def _clean_sim(sim_method: str) -> str:
    """Remove boilerplate from sim method name and replace underscores with spaces."""
    s = sim_method
    for token in _SIM_STRIP:
        s = s.replace(token, "")
    return s.replace("_", " ").strip()

def _abbreviate_label(tree_method: str, sim_method: str) -> str:
    tree_short = TREE_METHOD_ABBREVIATIONS.get(tree_method, tree_method)
    sim_short  = _clean_sim(sim_method)
    return f"tree={tree_short}  sim={sim_short}"


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

        with plt.rc_context(ANALYZER_RCPARAMS):
            fig, ax = plt.subplots(figsize=(3.5, 3.0))
            im = ax.imshow(data, aspect='equal', cmap="gray_r")
            plt.colorbar(im, ax=ax)

            for (r, c) in [start, goal]:
                rect = patches.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                ax.add_patch(rect)

            for r, c in path:
                ax.plot(c, r, 'o', markersize=6, color='blue',
                        markeredgecolor='black', markeredgewidth=1.2)

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

        with plt.rc_context(ANALYZER_RCPARAMS):
            fig, ax = plt.subplots(figsize=(3.5, 3.0))
            im = ax.imshow(data, cmap="gray_r", aspect="equal")
            plt.colorbar(im, ax=ax)

            for pos in [start] + ([goal] if goal else []):
                if pos is None:
                    continue
                r, c = pos
                rect = patches.Rectangle((c-0.5, r-0.5), 1, 1,
                                         linewidth=2, edgecolor='red',
                                         facecolor='none')
                ax.add_patch(rect)

            for r, c in path:
                ax.plot(c, r, 'o', color='blue', markersize=5,
                        markeredgecolor='black', markeredgewidth=1.0)

            for (r1, c1), (r2, c2) in zip(path[:-1], path[1:]):
                ax.arrow(c1, r1, c2 - c1, r2 - r1,
                         head_width=0.25, head_length=0.25,
                         fc='white', ec='black', linewidth=1.5,
                         length_includes_head=True, alpha=0.9)

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

        with plt.rc_context(ANALYZER_RCPARAMS):
            fig, ax = plt.subplots(figsize=(3.5, 3.0))
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
            "p": (-1, 0),
            "ö": (1,  0),
            "ä": (0,  1),
            "l": (0, -1),
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

        gr, gc = goal
        goal_patch = patches.Rectangle(
            (gc - 0.5, gr - 0.5), 1, 1,
            linewidth=2, edgecolor="#00ff88", facecolor="#00ff8840", zorder=3
        )
        ax.add_patch(goal_patch)
        ax.text(gc, gr, "G", ha="center", va="center",
                color="#00ff88", fontsize=9, fontweight="bold", zorder=4)

        sr, sc = start_pos
        start_patch = patches.Rectangle(
            (sc - 0.5, sr - 0.5), 1, 1,
            linewidth=2, edgecolor="#ffaa00", facecolor="#ffaa0030", zorder=3
        )
        ax.add_patch(start_patch)
        ax.text(sc, sr, "S", ha="center", va="center",
                color="#ffaa00", fontsize=9, fontweight="bold", zorder=4)

        r0, c0 = start_pos
        agent_dot, = ax.plot([c0], [r0], "o", color="#ff4466",
                             markersize=10, markeredgecolor="white",
                             markeredgewidth=1.5, zorder=5)

        path_line, = ax.plot([], [], "-", color="#ff446680",
                             linewidth=1.5, zorder=4)

        ax.set_xticks(range(ncols))
        ax.set_yticks(range(nrows))
        ax.tick_params(labelsize=6, colors="#aaaaaa")
        ax.grid(color="#333355", linestyle="-", linewidth=0.4)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

        stats_ax = fig.add_axes([0.74, 0.30, 0.24, 0.60])
        stats_ax.set_facecolor("#0d0d1a")
        stats_ax.set_xticks([])
        stats_ax.set_yticks([])
        for spine in stats_ax.spines.values():
            spine.set_edgecolor("#334466")

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
    #  MCTS TREE VISUALIZATION
    # ============================================================
    @staticmethod
    def visualize_mcts_svg(root, filename="mcts_tree.svg",
                           max_depth=None, show_metrics=True):
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

            label = f"D={getattr(node, '_depth', '?')}\\nV={getattr(node, '_visits', 0)}"
            if show_metrics:
                vals = getattr(node, '_values', {})
                if isinstance(vals, dict):
                    label += "\\n" + ", ".join(f"{k}:{float(v):.2f}" for k, v in vals.items())

            try:
                is_term = bool(node.is_terminal_state())
            except:
                is_term = False

            G.add_node(node_id, label=label, terminal=is_term)

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
    #  SAVE GIF
    # ============================================================
    @staticmethod
    def save_path_as_gif(environment: Environment,
                         start_pos: tuple[int, int],
                         moves: list,
                         gif_path="path.gif",
                         frame_duration=300):

        controller = Controller(environment, start_pos)

        with plt.rc_context(ANALYZER_RCPARAMS):
            fig, ax = plt.subplots(figsize=(3.5, 3.0))
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
        """Method that creates a heatmap of all maps in ./maps"""

        out_path = "./out"
        map_path = "./maps"

        if not os.path.exists(out_path):
            os.mkdir(out_path)

        if not os.path.exists(map_path):
            raise FileNotFoundError("Directory ./maps does not seem to exist.")

        with plt.rc_context(ANALYZER_RCPARAMS):
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
        with open(pickle_file, "rb") as f:
            env = pickle.load(f)

        filename = os.path.basename(pickle_file)
        name = os.path.splitext(filename)[0]
        name = re.sub(r"_\d+x\d+$", "", name)
        name = name.replace("_", " ")

        print(name)

        env = np.array(env)
        env_dim = env.shape[0]

        start = (0, env_dim // 2)
        goal  = (env_dim - 1, env_dim // 2)

        with plt.rc_context(ANALYZER_RCPARAMS):
            fig, ax = plt.subplots(figsize=(3.5, 3.0))

            im = ax.imshow(env, cmap="gray_r", origin="upper")

            start_square = Rectangle(
                (start[1] - 0.5, start[0] - 0.5), 1, 1,
                facecolor="green", edgecolor="black", linewidth=1.5
            )
            ax.add_patch(start_square)

            goal_square = Rectangle(
                (goal[1] - 0.5, goal[0] - 0.5), 1, 1,
                facecolor="red", edgecolor="black", linewidth=1.5
            )
            ax.add_patch(goal_square)

            legend_elements = [
                Patch(facecolor="green", edgecolor="black", label="Start"),
                Patch(facecolor="red",   edgecolor="black", label="Waypoint")
            ]
            ax.legend(handles=legend_elements, loc="upper right")

            ax.set_xticks(np.arange(-0.5, env_dim, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, env_dim, 1), minor=True)
            ax.grid(which="minor", color="black", linestyle="-", linewidth=0.3)
            ax.tick_params(which="minor", bottom=False, left=False)

            ax.set_title(name.title())

            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Obstacle Weight")

            plt.savefig(output_svg, format="svg", bbox_inches="tight")
            plt.close()

    def plot_pareto_maps(self) -> None:
        """Plot Pareto front for collected map dict — thesis-quality SVG output."""
        graphics_path = "./graphics/all_maps_mcts_wd/"
        os.makedirs(graphics_path, exist_ok=True)

        def pareto_filter(points):
            pareto = []
            for p in points:
                dominated = False
                for q in points:
                    if (
                        q["values"]["step_count"]    <= p["values"]["step_count"]
                        and q["values"]["weight_shifted"] <= p["values"]["weight_shifted"]
                    ) and (
                        q["values"]["step_count"]    < p["values"]["step_count"]
                        or q["values"]["weight_shifted"] < p["values"]["weight_shifted"]
                    ):
                        dominated = True
                        break
                if not dominated:
                    pareto.append(p)
            return pareto

        # Build global config set for consistent colour assignment across all maps
        all_configs = set()
        for points in self.map_container.values():
            for p in points:
                all_configs.add(
                    _abbreviate_label(p['tree_selection_method'], p['simulation_method'])
                )

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

            config_groups = defaultdict(lambda: {"steps": [], "weights": []})
            for p in points:
                label = _abbreviate_label(p['tree_selection_method'], p['simulation_method'])
                config_groups[label]["steps"].append(p["values"]["step_count"])
                config_groups[label]["weights"].append(p["values"]["weight_shifted"])

            with plt.rc_context(PARETO_PLOT_RCPARAMS):
                # Fixed landscape size: plot area left, compact legend right
                fig, ax = plt.subplots(figsize=(7.0, 3.5))

                for label, data in config_groups.items():
                    ax.scatter(
                        data["steps"], data["weights"],
                        marker="o", s=25,
                        label=label,
                        color=color_map[label],
                        linewidths=0.3,
                        edgecolors="white",
                        zorder=3,
                    )

                ax.set_xlabel("Step Count")
                ax.set_ylabel("Weight Shifted")
                ax.set_title(
                    f"Pareto Front — {map_name.replace('_', ' ').title()} {dim}×{dim}"
                )
                ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

                ax.legend(
                    title="Configuration",
                    bbox_to_anchor=(1.02, 1),
                    loc="upper left",
                    borderaxespad=0,
                    frameon=True,
                    framealpha=0.9,
                    edgecolor="#cccccc",
                    handlelength=1.0,
                    handletextpad=0.4,
                    borderpad=0.5,
                    labelspacing=0.3,
                ).get_title().set_fontweight("bold")

                plt.tight_layout()
                plt.savefig(
                    f"{graphics_path}/{map_name}-{dim}-front.svg",
                    format="svg",
                    bbox_inches="tight",
                )
                plt.close()

        # Console summary (keeps full method names for clarity)
        for (map_name, env_dim), points in self.map_container.items():
            if not points:
                continue

            filtered = pareto_filter(points)
            print(f"\n[{map_name} {env_dim}] total={len(points)} | pareto={len(filtered)}")

            config_groups = defaultdict(list)
            for p in filtered:
                config_key = (
                    f"{p['total_budget']}b/{p['per_sim_budget']}ps/"
                    f"{p['number_of_simulations']}n | "
                    f"tree={p['tree_selection_method']} | "
                    f"root={p['root_selection_method']} | "
                    f"sim={p['simulation_method']}"
                )
                config_groups[config_key].append(p)

            for config_key, config_points in config_groups.items():
                print(f"  [{len(config_points):>3} pts]  {config_key}")
                print(f"  Steps: {config_points[0]['values']['step_count']}, "
                      f"Weight Shifted: {config_points[0]['values']['weight_shifted']}")

    def rank_configurations(self, top_n: int = 3) -> None:
        """Ranks configurations by hypervolume of their own Pareto front, per map."""

        def pareto_filter_2d(points: list) -> list:
            pareto = []
            for p in points:
                if not any(
                    q[0] <= p[0] and q[1] <= p[1] and (q[0] < p[0] or q[1] < p[1])
                    for q in points
                ):
                    pareto.append(p)
            return pareto

        def hypervolume_2d(points: list, reference: tuple) -> float:
            if not points:
                return 0.0
            sorted_points = sorted(points, key=lambda p: p[0])
            ref_steps, ref_weight = reference
            hv = 0.0
            for i, (s, w) in enumerate(sorted_points):
                next_s = sorted_points[i + 1][0] if i + 1 < len(sorted_points) else ref_steps
                hv += (next_s - s) * (ref_weight - w)
            return hv

        for map_key, points in self.map_container.items():
            if not points:
                continue

            map_name, dim = map_key

            max_steps  = max(p["values"]["step_count"]    for p in points)
            max_weight = max(p["values"]["weight_shifted"] for p in points)
            min_steps  = min(p["values"]["step_count"]    for p in points)
            min_weight = min(p["values"]["weight_shifted"] for p in points)

            step_range   = max_steps  - min_steps  or 1
            weight_range = max_weight - min_weight or 1

            reference = (1.1, 1.1)
            ref_original_steps  = reference[0] * step_range  + min_steps
            ref_original_weight = reference[1] * weight_range + min_weight

            config_points = defaultdict(list)
            for p in points:
                label = (
                    f"tree={p['tree_selection_method']} | "
                    f"root={p['root_selection_method']} | "
                    f"sim={p['simulation_method']}"
                )
                norm_val = (
                    (p["values"]["step_count"]    - min_steps)  / step_range,
                    (p["values"]["weight_shifted"] - min_weight) / weight_range,
                )
                config_points[label].append(norm_val)

            config_hv = {
                label: hypervolume_2d(pareto_filter_2d(pts), reference)
                for label, pts in config_points.items()
            }

            ranked = sorted(config_hv.items(), key=lambda x: x[1], reverse=True)

            print(f"\n{'='*60}")
            print(f"  {map_name.replace('_', ' ').title()} {dim}x{dim}")
            print(f"  Steps  range: [{min_steps}, {max_steps}]")
            print(f"  Weight range: [{min_weight:.2f}, {max_weight:.2f}]")
            print(f"  Reference point (normalized):     {reference}")
            print(f"  Reference point (original scale): "
                  f"steps={ref_original_steps:.1f}, weight={ref_original_weight:.2f}")
            print(f"{'='*60}")

            for i, (label, hv) in enumerate(ranked[:top_n], 1):
                raw_pareto = pareto_filter_2d(config_points[label])
                raw_points = sorted(
                    [
                        (round(s * step_range + min_steps),
                         round(w * weight_range + min_weight, 2))
                        for s, w in raw_pareto
                    ],
                    key=lambda x: x[0]
                )
                print(f"\n  #{i}  HV={hv:.4f}")
                print(f"      {label}")

    def extract_pareto_front(self, dict_path: str) -> None:
        """Generates the Pareto front for the MCTS in form of images.

        Args:
            dict_path (str): Path to the result dictionary
        """

        def plot_pareto_fronts(datapoints: dict) -> None:
            for key, points in datapoints.items():
                if not points:
                    continue
                steps   = [p["values"]["step_count"]    for p in points]
                weights = [p["values"]["weight_shifted"] for p in points]
                with plt.rc_context(ANALYZER_RCPARAMS):
                    plt.figure(figsize=(3.5, 3.0))
                    plt.scatter(steps, weights, marker="o", label=key)
                    plt.xlabel("Step Count")
                    plt.ylabel("Weight Shifted")
                    plt.title("Pareto Front")
                    plt.grid(True)
                    plt.close()

        def pareto_filter(points):
            pareto = []
            for p in points:
                dominated = False
                for q in points:
                    if (
                        q["values"]["step_count"]    <= p["values"]["step_count"]
                        and q["values"]["weight_shifted"] <= p["values"]["weight_shifted"]
                    ) and (
                        q["values"]["step_count"]    < p["values"]["step_count"]
                        or q["values"]["weight_shifted"] < p["values"]["weight_shifted"]
                    ):
                        dominated = True
                        break
                if not dominated:
                    pareto.append(p)
            return pareto

        folder_name = os.path.basename(os.path.normpath(dict_path))
        map_match   = re.match(r'^([a-z]+(?:_[a-z]+)*_map)-(\d+)-(.+)$', folder_name)
        if map_match:
            folder_map_name = map_match.group(1)
            folder_map_dim  = int(map_match.group(2))
            folder_map_key  = (folder_map_name, folder_map_dim)
        else:
            folder_map_key  = (folder_name, -1)

        graphics_path = f"graphics/{dict_path.removeprefix('./')}"
        os.makedirs(graphics_path, exist_ok=True)
        datapoints = defaultdict(list)

        file_counts = defaultdict(lambda: defaultdict(lambda: {"none": 0, "total": 0}))

        for file in os.listdir(dict_path):
            with open(f"{dict_path}/{file}", "rb") as f:
                run_match        = re.search(r'-(\d+)\.pickle$', file)
                run_index        = int(run_match.group(1)) if run_match else -1
                new_filename     = re.sub(r'-\d+\.pickle$', '', file)
                data             = pickle.load(f)

                if data is not None:
                    for entry in data:
                        # Inject both run_index and the hyperparameter config key
                        # so test_configuration_significance can group correctly.
                        entry["run_index"]      = run_index
                        entry["hyperparam_key"] = new_filename
                        datapoints[new_filename].append(entry)
                        map_key = (entry["map_name"], entry["env_dim"])
                        file_counts[map_key][new_filename]["total"] += 1
                else:
                    file_counts[folder_map_key][new_filename]["none"]  += 1
                    file_counts[folder_map_key][new_filename]["total"] += 1

        for key in datapoints:
            for point in datapoints[key]:
                self.map_container[(point["map_name"], point["env_dim"])].append(point)

        print("\n=== None File Report ===")
        for map_key in sorted(file_counts.keys(), key=lambda x: str(x)):
            map_name, dim = map_key
            print(f"\n  {map_name.replace('_', ' ').title()} {dim}x{dim}")
            for config in sorted(file_counts[map_key].keys()):
                none_count = file_counts[map_key][config]["none"]
                if none_count > 0:
                    print(f"    {config}: {none_count} None")

        for key in datapoints:
            datapoints[key] = pareto_filter(datapoints[key])

        plot_pareto_fronts(datapoints=datapoints)


    def test_configuration_significance(self, alpha: float = 0.05) -> None:
        """Runs Kruskal-Wallis + Dunn post-hoc tests on per-run hypervolume scores
        across MCTS configurations, per map.

        Per-run HV is computed by:
        1. Grouping all points sharing the same (config_label, hyperparam_key, run_index),
            which uniquely identifies one algorithm run.
        2. Computing the Pareto front of that run's points.
        3. Computing the HV of that front against a shared normalized reference point.

        HV scores for all runs of the same config_label are then pooled into a
        distribution, which is what Kruskal-Wallis and Dunn operate on.

        Normalization uses the global min/max across all points for each map so
        that HV scores are directly comparable across configurations.

        Args:
            alpha (float): Significance level for all tests. Defaults to 0.05.
        """
        try:
            from scipy.stats import kruskal
            import scikit_posthocs as sp
        except ImportError:
            print("Please install scipy and scikit-posthocs: pip install scipy scikit-posthocs")
            return

        def pareto_filter_2d(points: list[tuple]) -> list[tuple]:
            """Returns the 2D Pareto front from a list of (step, weight) tuples."""
            pareto = []
            for p in points:
                if not any(
                    q[0] <= p[0] and q[1] <= p[1] and (q[0] < p[0] or q[1] < p[1])
                    for q in points
                ):
                    pareto.append(p)
            return pareto

        def hypervolume_2d(points: list[tuple], reference: tuple) -> float:
            """Computes the 2D hypervolume indicator relative to a reference point.

            Points must be dominated by the reference point (i.e. all normalized
            values must be below 1.1 on both axes) for the result to be meaningful.
            """
            if not points:
                return 0.0
            sorted_points = sorted(points, key=lambda p: p[0])
            ref_steps, ref_weight = reference
            hv = 0.0
            for i, (s, w) in enumerate(sorted_points):
                next_s = sorted_points[i + 1][0] if i + 1 < len(sorted_points) else ref_steps
                hv += (next_s - s) * (ref_weight - w)
            return hv

        for map_key, points in self.map_container.items():
            if not points:
                continue

            # Guard: run_index and hyperparam_key must have been injected by
            # extract_pareto_front. If missing, the patched version was not used.
            if "run_index" not in points[0] or "hyperparam_key" not in points[0]:
                print(
                    f"[WARNING] Points for {map_key} are missing 'run_index' or "
                    f"'hyperparam_key'.\n"
                    f"          Re-run load_all() with the patched extract_pareto_front()."
                )
                return

            map_name, dim = map_key
            print(f"\n{'='*60}")
            print(f"  {map_name.replace('_', ' ').title()} {dim}x{dim}")
            print(f"{'='*60}")

            # ------------------------------------------------------------------
            # Step 1: Compute global min/max across ALL points for this map so
            # that normalization is consistent across configurations. This ensures
            # that HV scores from different configs are on the same scale and
            # directly comparable.
            # ------------------------------------------------------------------
            all_steps   = [p["values"]["step_count"]    for p in points]
            all_weights = [p["values"]["weight_shifted"] for p in points]

            min_steps,  max_steps  = min(all_steps),  max(all_steps)
            min_weight, max_weight = min(all_weights), max(all_weights)

            step_range   = float(max_steps  - min_steps)  or 1.0
            weight_range = float(max_weight - min_weight) or 1.0

            # Normalized reference point: sits just outside the [0, 1] unit square
            # so that the best possible solution still contributes positive HV area.
            reference = (1.1, 1.1)

            print(f"  Steps  range : [{min_steps}, {max_steps}]")
            print(f"  Weight range : [{min_weight:.3f}, {max_weight:.3f}]")
            print(f"  Reference    : {reference} (normalized)")

            # ------------------------------------------------------------------
            # Step 2: Group normalized points by (config_label, hyperparam_key,
            # run_index). This triple uniquely identifies one algorithm run:
            #   - config_label   : tree selection + simulation method
            #   - hyperparam_key : specific hyperparameter combination
            #                      (budget, sim steps, n_simulations, ...)
            #   - run_index      : the repetition index (0..30 for 31 runs)
            #
            # A single pickle file maps to exactly one (hyperparam_key, run_index)
            # pair and may contain multiple Pareto points — one per solution found
            # during that run. We compute the per-run Pareto front and then its HV.
            # ------------------------------------------------------------------
            run_points: dict[tuple[str, str, int], list[tuple[float, float]]] = defaultdict(list)

            for p in points:
                config_label = _abbreviate_label(
                    p["tree_selection_method"],
                    p["simulation_method"],
                )
                run_key = (config_label, p["hyperparam_key"], p["run_index"])
                norm_point = (
                    (p["values"]["step_count"]    - min_steps)  / step_range,
                    (p["values"]["weight_shifted"] - min_weight) / weight_range,
                )
                run_points[run_key].append(norm_point)

            # ------------------------------------------------------------------
            # Step 3: Compute one HV scalar per unique run, then pool all HV
            # scalars belonging to the same config_label into a list. This gives
            # us the empirical HV distribution for each configuration, which is
            # what Kruskal-Wallis will compare.
            # ------------------------------------------------------------------
            config_hvs: dict[str, list[float]] = defaultdict(list)

            for (config_label, _hyperparam_key, _run_idx), raw_points in run_points.items():
                front  = pareto_filter_2d(raw_points)
                hv_val = hypervolume_2d(front, reference)
                config_hvs[config_label].append(hv_val)

            n_runs_per_config = {lbl: len(hvs) for lbl, hvs in config_hvs.items()}
            print(f"\n  Runs per configuration:")
            for lbl, n in sorted(n_runs_per_config.items()):
                print(f"    {lbl}: {n} runs")

            # ------------------------------------------------------------------
            # Step 4: Kruskal-Wallis omnibus test.
            # H0: all configurations have the same HV distribution.
            # We need at least 2 groups and each group needs at least 1 value.
            # ------------------------------------------------------------------
            labels = list(config_hvs.keys())
            groups = [config_hvs[lbl] for lbl in labels]

            if len(groups) < 2:
                print("  Fewer than 2 configurations — skipping.")
                continue

            stat, p_val = kruskal(*groups)
            print(f"\n  Kruskal-Wallis:  H = {stat:.4f},  p = {p_val:.6f}")

            if p_val >= alpha:
                print(f"  → No significant difference among configurations (p ≥ {alpha}).")
                continue

            print(f"  → Significant difference detected (p < {alpha}).")
            print(f"     Running Dunn post-hoc test with Bonferroni correction...")

            # ------------------------------------------------------------------
            # Step 5: Dunn post-hoc test with Bonferroni correction.
            # posthoc_dunn expects a list of 1-D array-like groups, one per config.
            # The returned DataFrame uses integer indices by default; we replace
            # them with the readable config labels.
            # ------------------------------------------------------------------
            dunn_matrix = sp.posthoc_dunn(groups, p_adjust="bonferroni")
            dunn_matrix.index   = labels
            dunn_matrix.columns = labels

            # Print the full p-value matrix with significance markers
            col_w = 14  # column width for p-values
            lbl_w = 45  # column width for row labels

            print(f"\n  Dunn post-hoc p-values (Bonferroni corrected)  (* = p < {alpha}):")
            print(f"  {'':>{lbl_w}s}", end="")
            for col in labels:
                print(f"  {col[:col_w]:>{col_w}s}", end="")
            print()

            for row_lbl in labels:
                print(f"  {row_lbl[:lbl_w]:<{lbl_w}s}", end="")
                for col_lbl in labels:
                    val    = dunn_matrix.loc[row_lbl, col_lbl]
                    marker = "*" if val < alpha and row_lbl != col_lbl else " "
                    print(f"  {val:>{col_w - 1}.4f}{marker}", end="")
                print()

            # Concise summary of significant pairs only
            print(f"\n  Significantly different pairs (p < {alpha} after correction):")
            found_any = False
            for i, l1 in enumerate(labels):
                for j, l2 in enumerate(labels):
                    if j <= i:
                        continue
                    val = dunn_matrix.loc[l1, l2]
                    if val < alpha:
                        print(f"    {l1}  vs  {l2}  →  p = {val:.6f}")
                        found_any = True
            if not found_any:
                print("    None after Bonferroni correction.")
    

    def load_all(self, log_path: str) -> None:
        """Loads all subdirectories in log_path into map_container.

        Args:
            log_path (str): Path to the root log directory
        """
        for dir in os.listdir(log_path):
            full_path = os.path.join(log_path, dir)
            if os.path.isdir(full_path):
                self.extract_pareto_front(full_path)

    def summarize_significance_tiers(self, alpha: float = 0.05) -> None:
        """
        Summarizes Kruskal-Wallis + Dunn results into interpretable performance
        tiers per map, based on which configurations are not significantly
        different from each other.

        Tiers are found via union-find: two configurations are placed in the
        same tier if their Dunn p-value >= alpha (i.e. no significant difference).
        Tiers are then ranked by their mean HV across all runs.

        Args:
            alpha (float): Significance level. Defaults to 0.05.
        """
        try:
            from scipy.stats import kruskal
            import scikit_posthocs as sp
        except ImportError:
            print("Please install scipy and scikit-posthocs: pip install scipy scikit-posthocs")
            return

        def pareto_filter_2d(points: list[tuple]) -> list[tuple]:
            pareto = []
            for p in points:
                if not any(
                    q[0] <= p[0] and q[1] <= p[1] and (q[0] < p[0] or q[1] < p[1])
                    for q in points
                ):
                    pareto.append(p)
            return pareto

        def hypervolume_2d(points: list[tuple], reference: tuple) -> float:
            if not points:
                return 0.0
            sorted_points = sorted(points, key=lambda p: p[0])
            ref_steps, ref_weight = reference
            hv = 0.0
            for i, (s, w) in enumerate(sorted_points):
                next_s = sorted_points[i + 1][0] if i + 1 < len(sorted_points) else ref_steps
                hv += (next_s - s) * (ref_weight - w)
            return hv

        def find(parent: dict, x: str) -> str:
            """Union-find path compression."""
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(parent: dict, rank: dict, x: str, y: str) -> None:
            """Union-find by rank."""
            rx, ry = find(parent, x), find(parent, y)
            if rx == ry:
                return
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1

        for map_key, points in self.map_container.items():
            if not points:
                continue

            if "run_index" not in points[0] or "hyperparam_key" not in points[0]:
                print(
                    f"[WARNING] Points for {map_key} are missing 'run_index' or "
                    f"'hyperparam_key'. Re-run load_all() with the patched "
                    f"extract_pareto_front()."
                )
                return

            map_name, dim = map_key

            # ------------------------------------------------------------------
            # Step 1: Normalize and compute per-run HV scores.
            # Identical to test_configuration_significance so results are consistent.
            # ------------------------------------------------------------------
            all_steps   = [p["values"]["step_count"]    for p in points]
            all_weights = [p["values"]["weight_shifted"] for p in points]

            min_steps,  max_steps  = min(all_steps),  max(all_steps)
            min_weight, max_weight = min(all_weights), max(all_weights)

            step_range   = float(max_steps  - min_steps)  or 1.0
            weight_range = float(max_weight - min_weight) or 1.0
            reference    = (1.1, 1.1)

            run_points: dict[tuple[str, str, int], list[tuple[float, float]]] = defaultdict(list)
            for p in points:
                config_label = _abbreviate_label(
                    p["tree_selection_method"],
                    p["simulation_method"],
                )
                run_key    = (config_label, p["hyperparam_key"], p["run_index"])
                norm_point = (
                    (p["values"]["step_count"]    - min_steps)  / step_range,
                    (p["values"]["weight_shifted"] - min_weight) / weight_range,
                )
                run_points[run_key].append(norm_point)

            config_hvs: dict[str, list[float]] = defaultdict(list)
            for (config_label, _, _), raw_points in run_points.items():
                front  = pareto_filter_2d(raw_points)
                hv_val = hypervolume_2d(front, reference)
                config_hvs[config_label].append(hv_val)

            labels = list(config_hvs.keys())
            groups = [config_hvs[lbl] for lbl in labels]

            if len(groups) < 2:
                continue

            # ------------------------------------------------------------------
            # Step 2: Kruskal-Wallis — skip tier analysis if not significant.
            # ------------------------------------------------------------------
            stat, p_val = kruskal(*groups)

            print(f"\n{'='*60}")
            print(f"  {map_name.replace('_', ' ').title()} {dim}x{dim}")
            print(f"{'='*60}")
            print(f"  Kruskal-Wallis:  H = {stat:.4f},  p < 0.001"
                if p_val < 0.001 else
                f"  Kruskal-Wallis:  H = {stat:.4f},  p = {p_val:.4f}")

            if p_val >= alpha:
                print(f"  → No significant difference among configurations (p ≥ {alpha}).")
                print(f"     All configurations form a single tier.")
                continue

            print(f"  → Significant difference detected — computing tiers...")

            # ------------------------------------------------------------------
            # Step 3: Run Dunn post-hoc and build tiers via union-find.
            # Two configs are merged into the same tier when p >= alpha,
            # meaning we cannot distinguish their performance.
            # ------------------------------------------------------------------
            dunn_matrix          = sp.posthoc_dunn(groups, p_adjust="bonferroni")
            dunn_matrix.index    = labels
            dunn_matrix.columns  = labels

            parent = {lbl: lbl for lbl in labels}
            rank   = {lbl: 0   for lbl in labels}

            for i, l1 in enumerate(labels):
                for j, l2 in enumerate(labels):
                    if j <= i:
                        continue
                    if dunn_matrix.loc[l1, l2] >= alpha:
                        union(parent, rank, l1, l2)

            # Group labels by their root in the union-find structure
            tier_groups: dict[str, list[str]] = defaultdict(list)
            for lbl in labels:
                tier_groups[find(parent, lbl)].append(lbl)

            # ------------------------------------------------------------------
            # Step 4: Rank tiers by mean HV (descending) and print summary.
            # ------------------------------------------------------------------
            tier_means = {
                root: np.mean([hv for lbl in members for hv in config_hvs[lbl]])
                for root, members in tier_groups.items()
            }
            ranked_tiers = sorted(tier_groups.items(),
                                key=lambda x: tier_means[x[0]], reverse=True)

            print(f"\n  Performance tiers ({len(ranked_tiers)} found):\n")

            for tier_rank, (root, members) in enumerate(ranked_tiers, 1):
                mean_hv  = tier_means[root]
                all_hvs  = [hv for lbl in members for hv in config_hvs[lbl]]
                std_hv   = float(np.std(all_hvs))
                min_hv   = float(np.min(all_hvs))
                max_hv   = float(np.max(all_hvs))

                # Decompose into unique tree / sim axes to spot patterns
                tree_methods = sorted({lbl.split("  ")[0] for lbl in members})
                sim_methods  = sorted({lbl.split("  ")[1] for lbl in members})

                print(f"  Tier {tier_rank}  (mean HV = {mean_hv:.4f} ± {std_hv:.4f}"
                    f",  range [{min_hv:.4f}, {max_hv:.4f}])")
                print(f"  {'─'*54}")

                for lbl in sorted(members):
                    n    = len(config_hvs[lbl])
                    m    = np.mean(config_hvs[lbl])
                    s    = np.std(config_hvs[lbl])
                    print(f"    {lbl:<40s}  n={n:>4}  HV={m:.4f} ± {s:.4f}")

                # Pattern summary: does the tree or sim method define this tier?
                if len(tree_methods) == 1:
                    print(f"\n    → Tree selection drives this tier: {tree_methods[0]}")
                if len(sim_methods) == 1:
                    print(f"    → Simulation method drives this tier: {sim_methods[0]}")
                if len(tree_methods) > 1 and len(sim_methods) > 1:
                    print(f"\n    → Tree methods : {', '.join(tree_methods)}")
                    print(f"       Sim methods  : {', '.join(sim_methods)}")
                print()

            # ------------------------------------------------------------------
            # Step 5: Cross-tier significance summary.
            # ------------------------------------------------------------------
            if len(ranked_tiers) > 1:
                print(f"  Cross-tier comparisons (all significant at p < 0.001):")
                for i, (r1, m1) in enumerate(ranked_tiers):
                    for r2, m2 in ranked_tiers[i + 1:]:
                        rep1 = m1[0]
                        rep2 = m2[0]
                        val  = dunn_matrix.loc[rep1, rep2]
                        p_str = "p < 0.001" if val < 0.001 else f"p = {val:.4f}"
                        print(f"    Tier {ranked_tiers.index((r1,m1))+1} vs "
                            f"Tier {ranked_tiers.index((r2,m2))+1}  →  {p_str}")