"""Multi-objective MCTS with a global Pareto archive.

Extends :class:`MctsTree` so the search targets the whole Pareto front
simultaneously. Terminal-solution objective vectors are pushed into a
tree-wide archive; selection strategies can then use HV / CD over this
global front instead of only a node-local archive.

Rollout strategies plug in via the same :mod:`strategies` registry, so
callers are free to swap in any existing rollout (light, square-sampling,
distance-weight) for experimentation.
"""

from __future__ import annotations
from typing import Optional

import copy
import random

import numpy as np

from mc_tree import MctsTree
from node import Node
from helper import Helper
from logger import Logger
from strategies.tree_selection import TreeSelectionStrategy
from strategies.root_selection import RootSelectionStrategy


class MOMctsTree(MctsTree):
    """MCTS variant that maintains a global Pareto archive of terminal paths."""

    def __init__(self, root: Node, seed: int, max_solutions: int = 20,
                 global_archive_size: int = 100,
                 progressive_widening_c: float = 1.5,
                 progressive_widening_alpha: float = 0.7) -> None:
        super().__init__(root, seed, max_solutions=max_solutions,
                         progressive_widening_c=progressive_widening_c,
                         progressive_widening_alpha=progressive_widening_alpha)
        # Each entry is ``(path, values)`` where ``path`` is a list of
        # ``((move_dir, shift_dir), value_dict)`` and ``values`` is the final
        # objective dict.
        self._global_archive: list[tuple[list, dict]] = []
        self._global_archive_size = global_archive_size

    # ------------------------------------------------------------------
    # Archive maintenance
    # ------------------------------------------------------------------

    @staticmethod
    def _dominates(a: dict, b: dict) -> bool:
        """True iff objective dict ``a`` Pareto-dominates ``b`` (minimization)."""
        not_worse = (
            a["step_count"] <= b["step_count"]
            and a["weight_shifted"] <= b["weight_shifted"]
            and a["distance_to_goal"] <= b["distance_to_goal"]
        )
        strictly_better = (
            a["step_count"] < b["step_count"]
            or a["weight_shifted"] < b["weight_shifted"]
            or a["distance_to_goal"] < b["distance_to_goal"]
        )
        return not_worse and strictly_better

    def add_to_global_archive(self, path: list, values: dict) -> None:
        """Insert a candidate solution into the global archive.

        Drops members dominated by the newcomer; skips the newcomer if it is
        dominated by any existing member. Truncates by HV contribution when
        the archive exceeds :attr:`_global_archive_size`.
        """
        for _, existing_vals in self._global_archive:
            if existing_vals == values:
                return
            if self._dominates(existing_vals, values):
                return

        self._global_archive = [
            entry for entry in self._global_archive
            if not self._dominates(values, entry[1])
        ]
        self._global_archive.append((copy.deepcopy(path), dict(values)))

        if len(self._global_archive) > self._global_archive_size:
            value_dicts = [entry[1] for entry in self._global_archive]
            contribs = Helper.hypervolume_contributions(value_dicts)
            order = sorted(range(len(contribs)), key=lambda i: contribs[i], reverse=True)
            keep = set(order[: self._global_archive_size])
            self._global_archive = [e for i, e in enumerate(self._global_archive) if i in keep]

    def global_pareto_values(self) -> list[dict]:
        """Return the current global Pareto front (value dicts only)."""
        return [entry[1] for entry in self._global_archive]

    # ------------------------------------------------------------------
    # Backprop hook: collect terminal objective vectors into the archive.
    # ------------------------------------------------------------------

    def backpropagate(self, node: Node, current_root: Node) -> None:  # type: ignore[override]
        # Capture the rollout terminal values before super().backpropagate
        # mutates ``node._values`` into running averages. The rollout strategy
        # writes ``node._values`` with the terminal objective vector reached
        # during simulation; that is what belongs in the global archive.
        rollout_values = dict(node._values) if node._values else None
        super().backpropagate(node, current_root)
        if rollout_values is None:
            return
        # The rollout always walks until its controller is terminal (either
        # goal reached or path exhausted), so ``rollout_values`` represents a
        # genuine terminal state and is always a valid Pareto candidate.
        path = self._reconstruct_path(node, current_root)
        self.add_to_global_archive(path, rollout_values)

    @staticmethod
    def _reconstruct_path(leaf: Node, current_root: Node) -> list:
        """Build a path of (move, value_dict) tuples from the root to the leaf."""
        path = []
        current = leaf
        root_parent = current_root._parent
        while current is not None and current is not root_parent:
            if current._last_move is not None:
                path.append((current._last_move, dict(current._values)))
            current = current._parent
        path.reverse()
        return path


# ----------------------------------------------------------------------
# Strategies that consult the global archive.
# ----------------------------------------------------------------------


class GlobalHVTreeSelection(TreeSelectionStrategy):
    """Tree-selection strategy: choose the child whose descendants contribute
    the most to the global Pareto archive's hypervolume.

    Falls back to the node-local HV selection when global archive data is
    unavailable for the current children.
    """

    name = "global_hv_tree_selection"

    def select(self, tree, node):
        return self._select_impl(tree, node)

    def _select_impl(self, tree, node: Node) -> Node:
        children = list(node._children.values())
        if not children:
            raise RuntimeError("GlobalHVTreeSelection called on node without children")

        archive_values = tree.global_pareto_values() if isinstance(tree, MOMctsTree) else []
        if not archive_values:
            return tree.pareto_path_child_selection_hv(node)

        child_values = [dict(child._values) for child in children]
        combined = child_values + archive_values
        contribs = Helper.hypervolume_contributions(combined)

        child_contribs = np.asarray(contribs[: len(child_values)], dtype=float)
        if np.sum(child_contribs) <= 1e-12:
            return random.choice(children)
        probs = child_contribs / np.sum(child_contribs)
        idx = np.random.choice(len(children), p=probs)
        return children[idx]


class GlobalHVRootSelection(RootSelectionStrategy):
    """Root-advancement strategy using global archive contributions."""

    name = "global_hv_root_selection"

    def select(self, tree, root):
        return self._select_impl(tree, root)

    def _select_impl(self, tree, root: Node) -> Node:
        pareto_children = Helper.determine_pareto_front_from_nodes(root._children.values())
        if not pareto_children:
            return random.choice(list(root._children.values()))
        if len(pareto_children) == 1:
            return pareto_children[0]

        archive_values = tree.global_pareto_values() if isinstance(tree, MOMctsTree) else []
        child_values = [dict(c._values) for c in pareto_children]
        combined = child_values + archive_values if archive_values else child_values
        contribs = Helper.hypervolume_contributions(combined)

        child_contribs = np.asarray(contribs[: len(child_values)], dtype=float)
        if np.sum(child_contribs) <= 1e-12:
            return random.choice(pareto_children)
        probs = child_contribs / np.sum(child_contribs)
        idx = np.random.choice(len(pareto_children), p=probs)
        return pareto_children[idx]
