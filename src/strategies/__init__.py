"""Strategy package for pluggable MCTS components.

Exposes strategy classes for tree selection, rollouts, and root selection,
plus integer-ID registries used for backward compatibility with the old
integer-dispatch API in ``simulations()``.
"""

from strategies.tree_selection import (
    TreeSelectionStrategy,
    UCBSelection,
    HVSelection,
    CDSelection,
    AEGASelection,
)
from strategies.rollout import (
    RolloutStrategy,
    LightRollout,
    SquareSamplingRollout,
    DistanceWeightRollout,
)
from strategies.root_selection import (
    RootSelectionStrategy,
    HVRootSelection,
)


TREE_SELECTION_REGISTRY = {
    0: UCBSelection,
    1: HVSelection,
    2: CDSelection,
    3: AEGASelection,
}

ROLLOUT_REGISTRY = {
    0: LightRollout,
    1: SquareSamplingRollout,
    2: DistanceWeightRollout,
}

def _global_hv_root():
    from mo_mc_tree import GlobalHVRootSelection
    return GlobalHVRootSelection


def _global_hv_tree():
    from mo_mc_tree import GlobalHVTreeSelection
    return GlobalHVTreeSelection


ROOT_SELECTION_REGISTRY = {
    0: HVRootSelection,
    1: _global_hv_root,
}

# Extend tree-selection registry with the global-archive variant lazily so the
# import of mo_mc_tree (which imports from this module) does not cycle.
TREE_SELECTION_REGISTRY[4] = _global_hv_tree


def build_tree_selection(indicator):
    """Return a tree-selection strategy from an integer indicator or instance."""
    if isinstance(indicator, TreeSelectionStrategy):
        return indicator
    try:
        cls = TREE_SELECTION_REGISTRY[indicator]
    except KeyError as exc:
        raise ValueError(f"Unknown tree selection indicator: {indicator}") from exc
    if callable(cls) and not isinstance(cls, type):
        cls = cls()
    return cls()


def build_rollout(indicator):
    """Return a rollout strategy from an integer indicator or instance."""
    if isinstance(indicator, RolloutStrategy):
        return indicator
    try:
        return ROLLOUT_REGISTRY[indicator]()
    except KeyError as exc:
        raise ValueError(f"Unknown rollout indicator: {indicator}") from exc


def build_root_selection(indicator):
    """Return a root-selection strategy from an integer indicator or instance."""
    if isinstance(indicator, RootSelectionStrategy):
        return indicator
    try:
        cls = ROOT_SELECTION_REGISTRY[indicator]
    except KeyError as exc:
        raise ValueError(f"Unknown root selection indicator: {indicator}") from exc
    if callable(cls) and not isinstance(cls, type):
        cls = cls()
    return cls()
