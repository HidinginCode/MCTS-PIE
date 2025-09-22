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

    def __init__(self, map_dim: int = 10) -> None:
        """Init function for the map class.

        Args:
            map_dim (int): Dimension of the quadratic man (map_dim x map_dim), default = 10
        """

        self.map_dim = map_dim
        self.map = [[random.random() for _ in range(map_dim)] for _ in range(map_dim)]
