"""Root-advancement strategies.

Each strategy chooses the next root from the current root's children when
advancing the tree after a search iteration. Implementations wrap the
original ``MctsTree`` methods.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from node import Node
    from mc_tree import MctsTree


class RootSelectionStrategy:
    """Base class for root-selection strategies."""

    name: str = "base"

    def select(self, tree: "MctsTree", root: "Node") -> "Node":
        raise NotImplementedError

    def __call__(self, root: "Node") -> "Node":
        return self.select(self._tree, root)  # type: ignore[attr-defined]

    def bind(self, tree: "MctsTree") -> "RootSelectionStrategy":
        self._tree = tree
        return self


class HVRootSelection(RootSelectionStrategy):
    name = "hv_root_selection"

    def select(self, tree, root):
        return tree.hv_root_selection(root)
