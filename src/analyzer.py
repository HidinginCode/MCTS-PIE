"""This module contains the analyzer class, used to log and create visual data."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
        data = environment
        highlighted_cells = [start, goal]
        fig, ax = plt.subplots()
        im = ax.imshow(data, aspect='equal',cmap="gray")


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

        plt.show()
