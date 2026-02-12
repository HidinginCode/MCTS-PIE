"""This module contains a helper class which contains some functions that might need to be called."""
from node import Node
import numpy as np
import pymoo.indicators.hv as HV

class Helper():
    """Helper class with helper methods."""

    @staticmethod
    def stable_normalize(values: list[dict[str, float]], eps: float = 1e-9) -> list[dict[str, float]]:
        """
        Robust multi-objective normalization for MCTS.
        Uses z-score normalization with safe handling of near-constant dimensions.

        Args:
            values: list of dicts, one per node, with keys:
                    "step_count", "weight_shifted", "distance_to_goal"
            eps: numerical stabilization constant

        Returns:
            normalized_list: list of dicts with same keys, but normalized values
        """

        # --- Extract values into matrix ---
        keys = list(values[0].keys())
        mat = np.array([[v[k] for k in keys] for v in values], dtype=float)

        # --- Compute means and stds ---
        means = mat.mean(axis=0)
        stds = mat.std(axis=0)

        # --- Avoid division by zero ---
        stds = np.where(stds < eps, 1.0, stds)

        # --- Z-score normalization ---
        normalized = (mat - means) / stds

        # Convert back to list-of-dicts
        normalized_list = [
            {k: float(normalized[i, j]) for j, k in enumerate(keys)}
            for i in range(len(values))
        ]

        return normalized_list
    
    @staticmethod
    def stable_minmax(matrix: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        """
        Safe [0,1] min-max normalization.
        Used in epsilon-clustering and crowding distance.
        """
        mins = matrix.min(axis=0)
        maxs = matrix.max(axis=0)
        ranges = maxs - mins
        ranges = np.where(ranges < eps, 1.0, ranges)

        return (matrix - mins) / ranges
        
    @staticmethod
    def determine_pareto_front_from_nodes(nodes, use_ucb_values=False):
        """
        FAST Pareto-front computation using Kung's O(n log n) skyline algorithm.
        Minimization for 3 objectives.
        """

        if not nodes:
            return []

        # Extract 3-tuple objective values for each node
        # MUCH faster than dict lookups inside dominance checks
        if use_ucb_values:
            vals = [(n, 
                    (n._ucb_values["step_count"],
                    n._ucb_values["weight_shifted"],
                    n._ucb_values["distance_to_goal"]))
                    for n in nodes]
        else:
            vals = [(n, 
                    (n._values["step_count"],
                    n._values["weight_shifted"],
                    n._values["distance_to_goal"]))
                    for n in nodes]

        # Sort lexicographically by objective 1, then 2, then 3
        vals.sort(key=lambda x: x[1])

        # Divide & conquer skyline computation
        def skyline(items):
            n = len(items)
            if n <= 1:
                return items

            left = skyline(items[: n//2])
            right = skyline(items[n//2 :])

            # Filter right items by checking dominance against left skyline
            filtered_right = []
            for node, (a1, a2, a3) in right:
                dominated = False
                for ln, (b1, b2, b3) in left:

                    # Branchless, inlined 3D dominance check
                    if (b1 <= a1 and b2 <= a2 and b3 <= a3) and \
                    (b1 < a1 or b2 < a2 or b3 < a3):
                        dominated = True
                        break

                if not dominated:
                    filtered_right.append((node, (a1, a2, a3)))

            # Merge left and filtered right
            return left + filtered_right

        # Run skyline on sorted items
        pf = skyline(vals)

        # Return nodes only
        return [n for n, _ in pf]

    
    @staticmethod
    def hypervolume(points: list) -> tuple:
        """Returns the index of the value with the biggest hypervolume for the list of points provided.

        Args:
            points (list): List of HV per point
        
        Returns:
            tuple: Hypervolumes and full HV as tuple
        """
        values = [list(point.values()) for point in points]
        values_norm = Helper.stable_minmax(np.array(values))
        #print(values)

        # Compute reference point
        worst = np.max(values_norm, axis=0)
        ranges = np.max(values_norm, axis=0) - np.min(values_norm, axis=0)
        ref = worst + 0.1 * (ranges + 1e-12) # Add margin to worst point

        hv = HV.Hypervolume(ref_point=np.array(ref))
        return ([hv.do((np.array(val))) for val in values_norm])

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

        # Use the robust min-max helper (handles tiny ranges safely)
        normalized_values = Helper.stable_minmax(values)

        # Rebuild list of dicts
        normalized_archive = [
            {k: float(v) for k, v in zip(keys, norm_row)}
            for norm_row in normalized_values
        ]
        return normalized_archive

    def epsilon_clustering_for_nodes(node: Node, eps: float = 1e-4, eps_steps=0.001):
        """Returns the child from the given node that survives the epsilon clustering.

        Args:
            node (Node): Node from which child is selected.
            eps (float, optional): Epsilon start value. Defaults to 1e-4.
            eps_steps (float, optional): Steps in which to increase epsilon. Defaults to 0.001.

        Returns:
            Node: Child that survived clustering
        """
        children = list(node._children.values())

        if not children:
            return None
        if len(children) == 1:
            return children[0]

        while len(children) > 1:
            # Normalize CURRENT children (recomputed each iteration)
            raw_value_dicts = [child._values for child in children]
            normalized_dicts = Helper.normalize_archive(raw_value_dicts)

            # cell -> (score, child)
            archive: dict[tuple[int, ...], tuple[float, Node]] = {}

            for child, norm_dict in zip(children, normalized_dicts):
                norm_vals = np.array(list(norm_dict.values()), dtype=float)

                # epsilon grid cell in normalized space
                cell = tuple(np.floor(norm_vals / eps).astype(int))

                # representative score: L2 norm of normalized objectives (minimization)
                score = float(np.linalg.norm(norm_vals))

                if cell not in archive or score < archive[cell][0]:
                    archive[cell] = (score, child)

            # keep only the representative child from each occupied cell
            children = [entry[1] for entry in archive.values()]
            eps += eps_steps   # coarsen grid gradually

        return children[0]

    @staticmethod
    def crowding_distance(points: list) -> list:
        """Method that calculates crowding distances for a list of points.

        Args:
            points (list): Points to calculate crowding distances.

        Returns:
            list: List of crowding distances.
        """
        front = np.array([[value for value in point.values()] for point in points], float)
        front = Helper.stable_minmax(front)
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

    @staticmethod
    def epsilon_clustering(node: Node, max_archive_size: int, eps: float = 1e-4, eps_steps: float = 0.001):
        """
        Epsilon-clustering for pareto path archives.

        - Uses stable min-max normalization internally (via normalize_archive)
        - Uses L2-norm of normalized objective vectors to choose cell representatives
        - Preserves all existing control flow and archive structure
        """

        current = list(node._pareto_paths)

        # Nothing to compress
        if len(current) <= max_archive_size:
            return

        while len(current) > max_archive_size:

            # Extract raw values from final entry of each path
            # (value dict is always at path[-1][1])
            raw_values = [path[-1][1] for path in current]

            # Normalize using your stabilized min-max
            normalized_dicts = Helper.normalize_archive(raw_values)

            # Cell → (score, path)
            archive = {}

            for path, norm_dict in zip(current, normalized_dicts):
                vec = np.array(list(norm_dict.values()), dtype=float)

                # Epsilon grid cell in normalized space
                cell = tuple(np.floor(vec / eps).astype(int))

                # Balanced representative selection:
                # Use L2 norm of normalized objective vector
                score = float(np.linalg.norm(vec))

                # Keep the one with lowest score (minimization)
                if cell not in archive or score < archive[cell][0]:
                    archive[cell] = (score, path)

            # Keep only the representative paths
            current = [entry[1] for entry in archive.values()]

            eps += eps_steps

        node._pareto_paths = current
