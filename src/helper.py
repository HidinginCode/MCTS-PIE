"""This module contains a helper class which contains some functions that might need to be called."""
from node import Node
import numpy as np
import pymoo.indicators.hv as HV

class Helper():
    """Helper class with helper methods."""

    @staticmethod
    def determine_pareto_front_from_nodes(node_list: list[Node], ucb_flag: bool = False) -> list[Node]:
        """Determines pareto front based on values dict of nodes.

        Args:
            node_list (list[Node]): List of nodes
            ucb_flag (bool): Flag that can be switched to used ucb values instead of normal ones for domination

        Returns:
            list[Node]: Pareto front
        """

        def is_dominated(node1: Node, node2: Node, ucb_flag: bool) -> bool:
            """Determines if node 1 is dominated by node 2

            Args:
                node1 (Node): Node to be checked if its dominated
                node2 (Node): Node to be check if it dominates
                ucb_flag (bool): Switch for ucb or normal value domination

            Returns:
                bool: Domination status
            """

            if not ucb_flag:
                return (
                    (node2._values["step_count"] <= node1._values["step_count"]) and
                    (node2._values["weight_shifted"] <= node1._values["weight_shifted"]) and
                    (node2._values["distance_to_goal"] <= node1._values["distance_to_goal"]) and
                    ((node2._values["step_count"] < node1._values["step_count"]) or
                    (node2._values["weight_shifted"] < node1._values["weight_shifted"]) or
                    (node2._values["distance_to_goal"] < node1._values["distance_to_goal"])
                    )
                )
            
            return (
                (node2._ucb_values["step_count"] <= node1._ucb_values["step_count"]) and
                (node2._ucb_values["weight_shifted"] <= node1._ucb_values["weight_shifted"]) and
                (node2._ucb_values["distance_to_goal"] <= node1._ucb_values["distance_to_goal"]) and
                ((node2._ucb_values["step_count"] < node1._ucb_values["step_count"]) or
                (node2._ucb_values["weight_shifted"] < node1._ucb_values["weight_shifted"]) or
                (node2._ucb_values["distance_to_goal"] < node1._ucb_values["distance_to_goal"])
                )
            )

        non_dominated_nodes = []

        for node1 in node_list:
            # Set domination flag false
            dominated = False

            for node2 in node_list:
                if node1 is node2:
                    continue

                if is_dominated(node1, node2, ucb_flag):
                    dominated = True
                    break
            if not dominated:
                non_dominated_nodes.append(node1)
        
        return non_dominated_nodes
    
    @staticmethod
    def hypervolume(points: list) -> tuple:
        """Returns the index of the value with the biggest hypervolume for the list of points provided.

        Args:
            points (list): List of HV per point
        
        Returns:
            tuple: Hypervolumes and full HV as tuple
        """
        values = [list(point.values()) for point in points]
        #print(values)

        # Compute reference point
        worst = np.max(values, axis=0)
        ranges = np.max(values, axis=0) - np.min(values, axis=0)
        ref = worst + 0.1 * (ranges + 1e-12) # Add margin to worst point

        hv = HV.Hypervolume(ref_point=np.array(ref))
        return ([hv.do((np.array(val))) for val in values], hv.do(np.array(values)))

    @staticmethod
    def normalize_archive(archive: list[dict]) -> list[dict]:
        """
        Normalizes all objective values across an archive so that
        each objective contributes equally to dominance comparisons.

        Args:
            archive (list[dict]): List of value dicts, e.g. from node._pareto_paths

        Returns:
            list[dict]: List of normalized dicts
        """
        # Collect all objective names
        keys = list(archive[0].keys())
        values = np.array([[d[k] for k in keys] for d in archive], dtype=float)

        # Compute global min/max per objective
        mins = values.min(axis=0)
        maxs = values.max(axis=0)
        ranges = np.where(maxs - mins == 0, 1.0, maxs - mins)  # avoid div by 0

        # Normalize to [0, 1]
        normalized_values = (values - mins) / ranges

        # Rebuild list of dicts
        normalized_archive = [
            {k: float(v) for k, v in zip(keys, norm)}
            for norm in normalized_values
        ]
        return normalized_archive

    @staticmethod
    def epsilon_clustering(node: Node, max_archive_size: int, eps: float = 1e-4, eps_max: float = 1, eps_steps = 0.001):
        """Method that uses epsilon clustering to prune the pareto path archive of a node.

        Args:
            node (Node): Node to prune
            max_archive_size (int): Desired archive size.
            eps (float, optional): Epsilon value. Defaults to 0.0.
            eps_max (float, optional): Maximum epsilon value. Defaults to 1.
            eps_steps (float, optional): Epsilon step size. Defaults to 0.01.
        """
        
        current = list(node._pareto_paths)
        iteration = 0
        value_dicts = [path[-1][1] for path in current]
        normalized_dicts = Helper.normalize_archive(value_dicts)

        while len(current) > max_archive_size and eps <= eps_max:
            # Normalize all path value dicts for fair scaling

            # Assign each path to epsilon grid cell
            archive = {}
            for path, norm_dict in zip(current, normalized_dicts):
                values = np.array(list(norm_dict.values()))
                cell = tuple(np.floor(values / eps).astype(int))

                if cell not in archive:
                    archive[cell] = path
                else:
                    # Keep the representative with smaller total objective sum
                    old_values = np.array(list(archive[cell][0][1].values()))
                    new_values = np.array(list(path[0][1].values()))
                    if np.sum(new_values) < np.sum(old_values):
                        archive[cell] = path

            current = list(archive.values())
            iteration += 1
            eps += eps_steps

        node._pareto_paths = current

    @staticmethod
    def crowding_distance(points: list) -> list:
        """Method that calculates crowding distances for a list of points.

        Args:
            points (list): Points to calculate crowding distances.

        Returns:
            list: List of crowding distances.
        """
        front = np.array([[value for value in point.values()] for point in points])
        N, M = front.shape
        distances = np.zeros(N)

        # Mask for extreme points
        inf_mask = np.zeros(N, dtype=bool)

        for m in range(M):
            idx = np.argsort(front[:, m])
            sorted_front = front[idx, m]

            # Mark boundary points
            inf_mask[idx[0]] = True
            inf_mask[idx[-1]] = True

            # Normalize objective values (avoid division by zero)
            min_m = sorted_front[0]
            max_m = sorted_front[-1]
            denom = max_m - min_m
            if denom == 0:
                continue

            # Compute normalized distance contribution
            diff = (sorted_front[2:] - sorted_front[:-2]) / denom

            # Add distances to inner points
            distances[idx[1:-1]] += diff


        distances[inf_mask] = np.inf
        return distances.tolist()


