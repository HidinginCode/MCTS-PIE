import heapq
from environment import Environment
import os
from controller import Controller
from environment import Environment
from analyzer import Analyzer
from heuristics import manhattan_chain
import random
import matplotlib.pyplot as plt
import numpy as np

PARETO_RCPARAMS = {
    "font.size":       9,
    "axes.titlesize":  11,
    "axes.labelsize":  9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
}

class A_Star:

    def __init__(self, map_name: str, env_dim: int, start: tuple, goal: tuple, checkpoints: list = None):
        """Init method for the a star class.

        Args:
            map_name (str): Name of the map as a string
            env_dim (int): Dimension of the map
            start (tuple): Start coordinate
            goal (tuple): Goal Coordinate (single-checkpoint backward compat)
            checkpoints (list, optional): Ordered list of checkpoints the agent
                must visit before returning to ``start``. When ``None`` defaults
                to ``[goal]`` (single-goal case).
        """
        self.map_name = map_name
        self.env_dim = env_dim
        self.start = start
        self.goal = goal
        self.checkpoints = [tuple(goal)] if checkpoints is None else [tuple(cp) for cp in checkpoints]

        self.env = Environment(
            map_type=self.map_name,
            env_dim=self.env_dim,
            start_pos=self.start,
            goal=self.goal,
            checkpoints=self.checkpoints,
        )
        self.pareto_values = []
        self.grid = self.env.environment


    # ------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------

    class Node:
        def __init__(self, pos: tuple, checkpoint_idx: int, steps: int, weight_sum: float, parent: "A_Star.Node" = None):
            """Init method for the node class used in A-Star.

            Args:
                pos (tuple): Position of the agent
                checkpoint_idx (int): Index of the next checkpoint to collect
                steps (int): Steps taken
                weight_sum (float): Weight shifted
                parent (Node, optional): Parent node. Defaults to None.
            """
            self.pos = pos
            self.checkpoint_idx = checkpoint_idx
            self.steps = steps
            self.weight_sum = weight_sum
            self.parent = parent

    def manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def valid_moves(self, pos):
        moves = [(1,0),(0,1),(-1,0),(0,-1)]
        result = []
        for dx, dy in moves:
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < self.env_dim and 0 <= ny < self.env_dim:
                result.append((nx, ny))
        return result


    # ------------------------------------------------------------
    # Phase 1: Compute minimal steps
    # ------------------------------------------------------------

    def compute_shortest_steps(self):

        open_list = []
        counter = 0
        n_checkpoints = len(self.checkpoints)

        heapq.heappush(open_list, (0, counter, self.start, 0, 0))

        closed = {}

        while open_list:

            _, _, pos, checkpoint_idx, steps = heapq.heappop(open_list)

            state = (pos, checkpoint_idx)

            if state in closed and closed[state] <= steps:
                continue

            closed[state] = steps

            if checkpoint_idx == n_checkpoints and pos == self.start:
                return steps

            for next_pos in self.valid_moves(pos):

                next_checkpoint_idx = checkpoint_idx
                if next_checkpoint_idx < n_checkpoints and next_pos == self.checkpoints[next_checkpoint_idx]:
                    next_checkpoint_idx += 1

                next_steps = steps + 1

                h = manhattan_chain(
                    next_pos,
                    self.checkpoints[next_checkpoint_idx:],
                    self.start,
                )

                counter += 1
                heapq.heappush(
                    open_list,
                    (next_steps + h, counter,
                     next_pos, next_checkpoint_idx, next_steps)
                )

        return None


    # ------------------------------------------------------------
    # Phase 2: Minimize weight under step constraint
    # ------------------------------------------------------------

    def minimize_weight_with_step_bound(self, step_limit):

        open_list = []
        counter = 0
        n_checkpoints = len(self.checkpoints)

        start_node = self.Node(self.start, 0, 0, 0, None)

        heapq.heappush(open_list, (0, counter, start_node))

        closed = {}

        while open_list:

            _, _, current = heapq.heappop(open_list)

            if current.steps > step_limit:
                continue

            remaining = step_limit - current.steps

            min_needed = manhattan_chain(
                current.pos,
                self.checkpoints[current.checkpoint_idx:],
                self.start,
            )

            if min_needed > remaining:
                continue

            state = (current.pos, current.checkpoint_idx, current.steps)

            if state in closed and closed[state] <= current.weight_sum:
                continue

            closed[state] = current.weight_sum

            if current.checkpoint_idx == n_checkpoints and current.pos == self.start:
                return current

            for next_pos in self.valid_moves(current.pos):

                next_checkpoint_idx = current.checkpoint_idx
                if next_checkpoint_idx < n_checkpoints and next_pos == self.checkpoints[next_checkpoint_idx]:
                    next_checkpoint_idx += 1

                next_steps = current.steps + 1
                cell_weight = self.grid[next_pos[0]][next_pos[1]]
                next_weight = current.weight_sum + cell_weight

                next_node = self.Node(
                    next_pos,
                    next_checkpoint_idx,
                    next_steps,
                    next_weight,
                    current
                )

                counter += 1
                heapq.heappush(open_list, (next_weight, counter, next_node))

        return None


    # ------------------------------------------------------------
    # Epsilon-constraint loop
    # ------------------------------------------------------------

    def epsilon_constraint_search(self):
        max_epsilon = self.env_dim * 3

        pareto = []

        L_star = self.compute_shortest_steps()
        print("Minimal steps:", L_star)

        for epsilon in range(max_epsilon + 1):

            step_limit = L_star + epsilon

            goal_node = self.minimize_weight_with_step_bound(step_limit)

            if goal_node is None:
                continue

            steps = goal_node.steps
            weight = goal_node.weight_sum
            path = self.reconstruct_path(goal_node)

            pareto.append((steps, weight, path))

            print(f"ε={epsilon} -> steps={steps}, weight={weight}")

        real_paths = []
        for p in pareto:
            real_path = self.shifting_optimization(p)
            if real_path is not None:
                real_paths.append(real_path)

        self.logging(real_paths)
        self.plot_pareto_front()

        return real_paths


    # ------------------------------------------------------------
    # Optimize shifting
    # ------------------------------------------------------------

    def shifting_optimization(self, solution: tuple) -> list:
        """Optimizes the shifts for the already optimized path.

        Args:
            solution (tuple): Solution from epsilon constraint search

        Returns:
            list: Moves and shifts
        """
        original_steps, original_weight, path = solution
        controller = Controller(environment=self.env, start_pos=self.start)

        formatted_path = []
        for i in range(len(path) - 1):
            current_pos = path[i]
            next_pos = path[i + 1]
            corresponding_move = (next_pos[0] - current_pos[0], next_pos[1] - current_pos[1])

            valid_moves = controller.get_all_valid_pairs()
            valid_shifts = [pair[1] for pair in valid_moves if pair[0] == corresponding_move]

            good_shifts = []
            if controller._environment._environment[next_pos[0]][next_pos[1]] != 0:
                for shift in valid_shifts:
                    shifting_pos = (next_pos[0] + shift[0], next_pos[1] + shift[1])
                    if shifting_pos not in path:
                        good_shifts.append(shift)
                if len(good_shifts) == 0:
                    return None
                formatted_path.append((corresponding_move, random.choice(good_shifts)))
            else:
                formatted_path.append((corresponding_move, random.choice(valid_shifts)))
            controller.move(formatted_path[-1][0], formatted_path[-1][1])

        return formatted_path


    def reconstruct_path(self, node: Node) -> list:
        """Function that reconstructs the taken path from the goal node.

        Args:
            node (Node): Goal node

        Returns:
            list: Visited positions
        """
        path = []
        while node is not None:
            path.append(node.pos)
            node = node.parent
        path.reverse()
        return path


    def pareto_filter_with_paths(self, points):
        """Filters points and returns non-dominated (f1, f2, path) without duplicates."""
        unique = {}
        for f1, f2, path in points:
            key = (f1, f2, tuple(path))
            unique[key] = (f1, f2, path)

        points = list(unique.values())
        points.sort(key=lambda x: x[0])

        pareto = []
        best_f2 = float("inf")

        for f1, f2, path in points:
            if f2 < best_f2:
                pareto.append((f1, f2, path))
                best_f2 = f2

        self.pareto_values = pareto


######################################
#   Logging and Plotting
######################################

    def logging(self, solutions: list) -> None:
        """Logging function for found solutions.

        Args:
            solutions (list): Found solutions
        """
        log_path = f"./a_star_log/{self.map_name}-{self.env_dim}"
        gif_path = log_path + "/gifs"
        os.makedirs(log_path, exist_ok=True)
        os.makedirs(gif_path, exist_ok=True)

        tuples = []
        for solution in solutions:
            controller = Controller(environment=self.env, start_pos=self.start)
            for step in solution:
                move, shift = step
                controller.move(move, shift)
            tuples.append((controller.step_count, controller.weight_shifted, solution))

        for steps, weight, _ in tuples:
            print(f"Steps: {steps}, Weight: {weight}")

        self.pareto_filter_with_paths(tuples)

        for i, pareto_sol in enumerate(self.pareto_values):
            print(f"{i}: {pareto_sol[0]}, {pareto_sol[1]}")
            self.plot_path(pareto_sol[-1], filename=f"path-{i}.svg")


    def plot_pareto_front(self) -> None:
        """Plots the Pareto front of the A* epsilon-constraint search as an SVG."""
        if not self.pareto_values:
            print("No Pareto values to plot. Run epsilon_constraint_search() first.")
            return

        graphics_path = f"./graphics/all_maps-a_star"
        os.makedirs(graphics_path, exist_ok=True)

        steps   = [p[0] for p in self.pareto_values]
        weights = [p[1] for p in self.pareto_values]

        with plt.rc_context(PARETO_RCPARAMS):
            fig, ax = plt.subplots(figsize=(4.5, 3.8))
            ax.scatter(steps, weights, marker="o", color="steelblue", zorder=3)
            ax.set_xlabel("Step Count")
            ax.set_ylabel("Weight Shifted")

            if len(self.pareto_values) != 1:
                ax.set_title(
                    f"Pareto Front - {self.map_name.replace('_', ' ').title()} {self.env_dim}x{self.env_dim}"
                )
            else:
                ax.set_title(
                    f"Pareto Optimal Solution - {self.map_name.replace('_', ' ').title()} {self.env_dim}x{self.env_dim}"
                )

            ax.grid(True)
            plt.tight_layout()
            out_path = f"{graphics_path}/{self.map_name}-{self.env_dim}.svg"
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()
        print(f"Saved: {out_path}")


    def plot_path(self, solution: list, filename: str = "path.svg") -> None:
        """Visualizes a solution path on the map as an SVG.

        Executes all (move, shift) pairs via a fresh controller so the
        displayed grid reflects the actual modified environment after the
        run - weights that were shifted appear in their new positions.

        Args:
            solution (list): List of (move, shift) tuples from shifting_optimization
            filename (str): Output filename
        """
        graphics_path = f"./a_star_log/{self.map_name}-{self.env_dim}"
        os.makedirs(graphics_path, exist_ok=True)

        # Fresh controller - each plot is independent and does not pollute self.env
        controller = Controller(environment=self.env, start_pos=self.start)
        visited = [self.start]
        for move, shift in solution:
            controller.move(move, shift)
            visited.append(controller.current_pos)

        # Capture the grid AFTER all moves so shifted weights are visible
        grid = np.array(controller._environment._environment)

        with plt.rc_context(PARETO_RCPARAMS):
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(grid, cmap="gray_r", origin="upper")

            path_rows = [p[0] for p in visited]
            path_cols = [p[1] for p in visited]
            ax.plot(path_cols, path_rows, color="orange", linewidth=3, zorder=2)

            ax.scatter(self.start[1], self.start[0], color="lime", s=60, zorder=3, label="Start")
            ax.scatter(self.goal[1],  self.goal[0],  color="red",  s=60, zorder=3, label="Waypoint")

            ax.set_title(
                f"{self.map_name.replace('_', ' ').title()} {self.env_dim}x{self.env_dim}\n"
                f"Steps: {controller.step_count}  |  Weight Shifted: {controller.weight_shifted:.2f}"
            )
            ax.legend(loc="upper right")
            ax.set_xlabel("X-Axis")
            ax.set_ylabel("Y-Axis")
            plt.colorbar(im, ax=ax, label="Cell Weight")
            plt.tight_layout()
            out_path = f"{graphics_path}/{filename}"
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()
        print(f"Saved: {out_path}")


    def plot_pareto_comparison(self, analyzer: Analyzer) -> None:
        """Compares the A* Pareto front against the MCTS global Pareto front graphically.

        Args:
            analyzer (Analyzer): Analyzer instance with loaded map_container
        """
        if not self.pareto_values:
            print("No A* Pareto values to plot. Run epsilon_constraint_search() first.")
            return

        map_key     = (self.map_name, self.env_dim)
        mcts_points = analyzer.map_container.get(map_key, [])

        if not mcts_points:
            print(f"No MCTS data found for {self.map_name} {self.env_dim}x{self.env_dim}.")
            return

        def pareto_filter_mcts(points):
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

        mcts_pareto  = pareto_filter_mcts(mcts_points)
        mcts_sorted  = sorted(mcts_pareto, key=lambda p: p["values"]["step_count"])
        mcts_steps   = [p["values"]["step_count"]    for p in mcts_sorted]
        mcts_weights = [p["values"]["weight_shifted"] for p in mcts_sorted]

        astar_steps   = [p[0] for p in self.pareto_values]
        astar_weights = [p[1] for p in self.pareto_values]

        graphics_path = f"./graphics/comparison/"
        os.makedirs(graphics_path, exist_ok=True)

        with plt.rc_context(PARETO_RCPARAMS):
            fig, ax = plt.subplots(figsize=(4.5, 3.8))

            ax.scatter(mcts_steps,   mcts_weights,   marker="o", color="steelblue",
                       zorder=3, label=f"MCTS ({len(mcts_pareto)} pts)")
            ax.scatter(astar_steps,  astar_weights,  marker="o", color="crimson",
                       zorder=3, label=f"A* ({len(self.pareto_values)} pts)")

            ax.set_xlabel("Step Count")
            ax.set_ylabel("Weight Shifted")
            ax.set_title(
                f"Pareto Comparison - "
                f"{self.map_name.replace('_', ' ').title()} {self.env_dim}x{self.env_dim}"
            )
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            out_path = f"{graphics_path}/{self.map_name}-{self.env_dim}.svg"
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()
        print(f"Saved: {out_path}")


    @staticmethod
    def run_all(analyzer: Analyzer, goals: dict) -> None:
        """Runs A* for all map/dim combinations found in the analyzer and plots comparisons.

        Args:
            analyzer (Analyzer): Analyzer instance with loaded map_container
            goals (dict): { (map_name, env_dim): (goal_row, goal_col) }
        """
        for (map_name, env_dim) in analyzer.map_container.keys():
            start = (0, env_dim // 2)
            goal  = goals.get((map_name, env_dim), (env_dim - 1, env_dim // 2))

            print(f"\n{'='*60}")
            print(f"  Running A* on {map_name.replace('_', ' ').title()} {env_dim}x{env_dim}")
            print(f"  Start: {start}  |  Goal: {goal}")
            print(f"{'='*60}")

            astar = A_Star(
                map_name=map_name,
                env_dim=env_dim,
                start=start,
                goal=goal
            )
            astar.epsilon_constraint_search()
            astar.plot_pareto_comparison(analyzer)