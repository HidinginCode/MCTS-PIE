"""Rollout strategies.

Each strategy simulates from a leaf node and returns the number of moves used.
Implementations here wrap the original methods on ``MctsTree`` to keep
behavior identical during the Phase A refactor.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from node import Node
    from mc_tree import MctsTree


class RolloutStrategy:
    """Base class for rollout strategies."""

    name: str = "base"

    def rollout(
        self,
        tree: "MctsTree",
        leaf: "Node",
        simulations: int,
        maximum_moves: int,
        remaining_budget: int,
    ) -> int:
        raise NotImplementedError

    def __call__(
        self,
        leaf: "Node",
        simulations: int,
        maximum_moves: int,
        remaining_budget: int,
    ) -> int:
        return self.rollout(self._tree, leaf, simulations, maximum_moves, remaining_budget)  # type: ignore[attr-defined]

    def bind(self, tree: "MctsTree") -> "RolloutStrategy":
        self._tree = tree
        return self


class LightRollout(RolloutStrategy):
    name = "light_rollout"

    def rollout(self, tree, leaf, simulations, maximum_moves, remaining_budget):
        return tree.light_rollout(leaf, simulations, maximum_moves, remaining_budget)


class SquareSamplingRollout(RolloutStrategy):
    name = "iterative_heavy_square_sampling_rollout"

    def rollout(self, tree, leaf, simulations, maximum_moves, remaining_budget):
        return tree.iterative_heavy_square_sampling_rollout(
            leaf, simulations, maximum_moves, remaining_budget
        )


class DistanceWeightRollout(RolloutStrategy):
    name = "iterative_heavy_distance_weight_rollout"

    def rollout(self, tree, leaf, simulations, maximum_moves, remaining_budget):
        return tree.iterative_heavy_distance_weight_rollout(
            leaf, simulations, maximum_moves, remaining_budget
        )
