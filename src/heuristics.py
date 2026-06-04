"""Shared distance heuristics for A* and MCTS.

Centralizes the Manhattan-chain lower bound used for multi-checkpoint
scenarios so A* and the Controller stay consistent.
"""

from __future__ import annotations


def manhattan(a: tuple, b: tuple) -> int:
    """Manhattan distance between two grid positions."""
    ax, ay = a
    bx, by = b
    dx = ax - bx
    dy = ay - by
    return (dx if dx >= 0 else -dx) + (dy if dy >= 0 else -dy)


def manhattan_chain(pos: tuple,
                    remaining_checkpoints: list[tuple],
                    start: tuple) -> int:
    """Lower bound on remaining moves for a fixed-order multi-checkpoint tour.

    The agent must visit each checkpoint in ``remaining_checkpoints`` in
    order and then return to ``start``. The Manhattan metric gives an
    admissible lower bound on the number of cardinal moves required.

    If no checkpoints remain, returns the Manhattan distance from ``pos``
    back to ``start`` (i.e. the return-home leg).
    """
    if not remaining_checkpoints:
        return manhattan(pos, start)

    total = manhattan(pos, remaining_checkpoints[0])
    for i in range(len(remaining_checkpoints) - 1):
        total += manhattan(remaining_checkpoints[i], remaining_checkpoints[i + 1])
    total += manhattan(remaining_checkpoints[-1], start)
    return total
