"""Map module.

This module defines the `Map` class, which represents a 2D grid of obstacles
with values between 0 and 1 (density). The map is initialized with random
values in this range.
"""

import random

class Map():
    """This class holds the map object.
    A map consists of a two dimensional list, that holds obstacles of a certain density.

    The density is always between '0' and '1'.

    Obstacles can be combined, but only if they stay in the specified density interval.
    """

    # Global Arguments for Map Class
    map = []
    map_dim = 0
    identificator = None

    def __init__(self, map_dim: int = 10) -> None:
        """Init function for the map class.

        Args:
            map_dim (int): Dimension of the quadratic man (map_dim x map_dim), default = 10
        """

        self.map_dim = map_dim
        self.map = [[random.random() for _ in range(map_dim)] for _ in range(map_dim)]
        self.identificator = id(self)

    def get_map(self) -> list:
        """ Returns the map list.

        Returns:
            list: 2 dimensional map list.
        """
        return self.map

    def get_map_dim(self) -> int:
        """ Returns the map dimensions.

        Returns:
            int: Map dimension
        """
        return self.map_dim

    def get_identificator(self) -> int:
        """Returns map identificator.

        Returns:
            int: Map identificator
        """
        return self.identificator
