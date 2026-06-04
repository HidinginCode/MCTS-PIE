"""Tree-selection strategies.

Each strategy takes a tree and a node and returns the child node to descend
into. Implementations here wrap the original methods on ``MctsTree`` so
behavior is byte-identical to the pre-refactor codebase.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from node import Node
    from mc_tree import MctsTree


class TreeSelectionStrategy:
    """Base class for tree-selection strategies."""

    name: str = "base"

    def select(self, tree: "MctsTree", node: "Node") -> "Node":
        raise NotImplementedError

    def __call__(self, node: "Node") -> "Node":
        # Allows instances to be bound to a tree and used as the legacy callable
        # (the bound tree is attached via ``bind``).
        return self.select(self._tree, node)  # type: ignore[attr-defined]

    def bind(self, tree: "MctsTree") -> "TreeSelectionStrategy":
        self._tree = tree
        return self


class UCBSelection(TreeSelectionStrategy):
    name = "ucb_child_selection"

    def select(self, tree, node):
        return tree.ucb_child_selection(node)


class HVSelection(TreeSelectionStrategy):
    name = "pareto_path_child_selection_hv"

    def select(self, tree, node):
        return tree.pareto_path_child_selection_hv(node)


class CDSelection(TreeSelectionStrategy):
    name = "pareto_path_child_selection_cd"

    def select(self, tree, node):
        return tree.pareto_path_child_selection_cd(node)


class AEGASelection(TreeSelectionStrategy):
    name = "pareto_path_child_selection_aega"

    def select(self, tree, node):
        return tree.pareto_path_child_selection_aega(node)
