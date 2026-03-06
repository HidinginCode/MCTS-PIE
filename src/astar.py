import heapq
from environment import Environment
import os
from controller import Controller
from environment import Environment
from analyzer import Analyzer
import random

class A_Star:

    def __init__(self, map_name: str, env_dim: int, start: tuple, goal: tuple):
        self.map_name = map_name
        self.env_dim = env_dim
        self.start = start
        self.goal = goal

        self.env = Environment(
            map_type=self.map_name,
            env_dim=self.env_dim,
            start_pos=self.start,
            goal=self.goal
        )
        self.pareto_values = []
        self.grid = self.env.environment


    # ------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------

    class Node:
        def __init__(self, pos, goal_collected, steps, weight_sum, parent=None):
            self.pos = pos
            self.goal_collected = goal_collected
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
    # Phase A: Compute minimal steps (pure geometry)
    # ------------------------------------------------------------

    def compute_shortest_steps(self):

        open_list = []
        counter = 0

        heapq.heappush(open_list, (0, counter, self.start, False, 0))

        closed = {}

        while open_list:

            _, _, pos, goal_collected, steps = heapq.heappop(open_list)

            state = (pos, goal_collected)

            if state in closed and closed[state] <= steps:
                continue

            closed[state] = steps

            if goal_collected and pos == self.start:
                return steps

            for next_pos in self.valid_moves(pos):

                next_goal_collected = goal_collected
                if next_pos == self.goal:
                    next_goal_collected = True

                next_steps = steps + 1

                # heuristic
                if not next_goal_collected:
                    h = self.manhattan(next_pos, self.goal) + \
                        self.manhattan(self.goal, self.start)
                else:
                    h = self.manhattan(next_pos, self.start)

                counter += 1
                heapq.heappush(
                    open_list,
                    (next_steps + h, counter,
                     next_pos, next_goal_collected, next_steps)
                )

        return None


    # ------------------------------------------------------------
    # Phase B: Minimize weight under step constraint
    # ------------------------------------------------------------

    def minimize_weight_with_step_bound(self, step_limit):

        open_list = []
        counter = 0

        start_node = self.Node(self.start, False, 0, 0, None)

        heapq.heappush(open_list,
                    (0, counter, start_node))

        closed = {}

        while open_list:

            _, _, current = heapq.heappop(open_list)

            if current.steps > step_limit:
                continue

            remaining = step_limit - current.steps

            if not current.goal_collected:
                min_needed = self.manhattan(current.pos, self.goal) + \
                            self.manhattan(self.goal, self.start)
            else:
                min_needed = self.manhattan(current.pos, self.start)

            if min_needed > remaining:
                continue

            state = (current.pos, current.goal_collected, current.steps)

            if state in closed and closed[state] <= current.weight_sum:
                continue

            closed[state] = current.weight_sum

            if current.goal_collected and current.pos == self.start:
                return current  # return full node

            for next_pos in self.valid_moves(current.pos):

                next_goal_collected = current.goal_collected
                if next_pos == self.goal:
                    next_goal_collected = True

                next_steps = current.steps + 1
                cell_weight = self.grid[next_pos[0]][next_pos[1]]
                next_weight = current.weight_sum + cell_weight

                next_node = self.Node(
                    next_pos,
                    next_goal_collected,
                    next_steps,
                    next_weight,
                    current
                )

                counter += 1
                heapq.heappush(open_list,
                            (next_weight, counter, next_node))

        return None


    # ------------------------------------------------------------
    # Epsilon-constraint loop
    # ------------------------------------------------------------

    def epsilon_constraint_search(self):
        max_epsilon = self.env_dim*2+30

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

        #self.logging(pareto)
        real_paths = []
        for p in pareto:
            real_path = self.shifting_optimization(p)
            if real_path is not None:
                real_paths.append(real_path)
        
        self.logging(real_paths)

        return real_paths
    
    #-------------------------------------------------------------
    # Optimize shifting
    #-------------------------------------------------------------
    def shifting_optimization(self, solution: tuple) -> list:
        """Optimizes the shifts for the already optimized path

        Args:
            solution (tuple): Solution from epsilon constraint search

        Returns:
            list: Moves and shifts
        """

        original_steps, original_weight, path = solution
        controller = Controller(environment=self.env, start_pos=self.start)

        formatted_path = []
        for i in range(len(path)-1):
            current_pos = path[i]
            next_pos = path[i+1]
            corresponding_move = (next_pos[0]-current_pos[0], next_pos[1]-current_pos[1])
            
            valid_moves = controller.get_all_valid_pairs()
            valid_shifts = [pair[1] for pair in valid_moves if pair[0] == corresponding_move]
            
            good_shifts = []
            if controller._environment._environment[next_pos[0]][next_pos[1]] != 0:
                for shift in valid_shifts:
                    shifting_pos = (next_pos[0]+shift[0], next_pos[1]+shift[1])
                    print(next_pos)
                    print(shifting_pos)
                    print(path.count(shifting_pos))
                    if shifting_pos not in path:
                        good_shifts.append(shift)
                if len(good_shifts) == 0:
                    formatted_path = None
                    return formatted_path
                formatted_path.append((corresponding_move, random.choice(good_shifts)))
            else:
                formatted_path.append((corresponding_move, random.choice(valid_shifts)))
            controller.move(formatted_path[-1][0], formatted_path[-1][1])
        return formatted_path

    
    def reconstruct_path(self, node):

        path = []

        while node is not None:
            path.append(node.pos)
            node = node.parent

        path.reverse()
        return path

    def pareto_filter_with_paths(self, points):
        """
        points: list of (f1, f2, path)

        Returns:
            list of non-dominated (f1, f2, path)
            without exact duplicates.
        """

        # Remove exact duplicates first
        unique = {}
        for f1, f2, path in points:
            key = (f1, f2, tuple(path))
            unique[key] = (f1, f2, path)

        points = list(unique.values())

        # Sort by first objective
        points.sort(key=lambda x: x[0])

        pareto = []
        best_f2 = float("inf")

        for f1, f2, path in points:
            if f2 < best_f2:
                pareto.append((f1, f2, path))
                best_f2 = f2

        self.pareto_values = pareto

    def logging(self, solutions: list) -> None:
        """Logging function for found solution.

        Args:
            solutions (list): Found solutions
        """

        log_path = f"./a_star_log/{self.map_name}-{self.env_dim}"
        gif_path = log_path+"/gifs"
        os.makedirs(log_path, exist_ok = True)
        os.makedirs(gif_path, exist_ok = True)
        
        tuples = []
        for i, solution in enumerate(solutions):
            controller = Controller(environment=self.env, start_pos=self.start)
            for step in solution:
                move, shift = step
                controller.move(move, shift)
            tuples.append((controller.step_count, controller.weight_shifted, solution))
        
        self.pareto_filter_with_paths(tuples)
        print(self.pareto_values)
        for i, pareto_sol in enumerate(self.pareto_values):
            Analyzer.save_path_as_gif(self.env, self.start, pareto_sol[-1], f"{gif_path}/solution-{i}.gif")