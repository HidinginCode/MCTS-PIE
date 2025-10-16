"""This module contains the direction enum."""

from enum import Enum

class Direction (Enum):
    """This class represents the direction enum."""

    NORTH = [0,1]
    SOUTH = [0,-1]
    EAST = [1,0]
    WEST = [-1,0]
