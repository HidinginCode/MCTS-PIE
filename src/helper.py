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
                    (node1._values["step_count"] <= node2._values["step_count"]) and
                    (node1._values["weight_shifted"] <= node2._values["weight_shifted"]) and
                    (node1._values["distance_to_goal"] <= node2._values["distance_to_goal"]) and
                    ((node1._values["step_count"] < node2._values["step_count"]) or
                    (node1._values["weight_shifted"] < node2._values["weight_shifted"]) or
                    (node1._values["distance_to_goal"] < node2._values["distance_to_goal"])
                    )
                )
            
            return (
                (node1._ucb_values["step_count"] <= node2._ucb_values["step_count"]) and
                (node1._ucb_values["weight_shifted"] <= node2._ucb_values["weight_shifted"]) and
                (node1._ucb_values["distance_to_goal"] <= node2._ucb_values["distance_to_goal"]) and
                ((node1._ucb_values["step_count"] < node2._ucb_values["step_count"]) or
                (node1._ucb_values["weight_shifted"] < node2._ucb_values["weight_shifted"]) or
                (node1._ucb_values["distance_to_goal"] < node2._ucb_values["distance_to_goal"])
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

