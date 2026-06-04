"""Run specs, shared seed list, and the 10-map suite for the paper protocol."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


# 30 seeds, hard-coded so every family is forced to use the same set.
SEEDS: list[int] = list(range(1000, 1030))

# 5 map types x 2 sizes = 10 maps. The 50x50 size is reserved for E4 scaling.
MAP_TYPES = [
    "easy_map",
    "checkerboard_map",
    "random_map",
    "meandering_river_map",
    "random_maze",
]
MAP_SIZES = [20, 35]
MAP_SUITE: list[tuple[str, int]] = [(m, d) for d in MAP_SIZES for m in MAP_TYPES]

# All algorithms compared in E1.
ALGORITHMS = (
    "mcts",
    "mo_mcts",
    "two_phase_mcts",
    "two_phase_momcts",
    "astar_mcts",
    "astar",
)

# Default budgets for the main protocol.
DEFAULT_TOTAL_BUDGET = 4000
DEFAULT_PER_SIM_BUDGET = 20
DEFAULT_SIMS_PER_CHILD = 20


def default_start_goal(env_dim: int) -> tuple[tuple[int, int], tuple[int, int]]:
    return (0, env_dim // 2), (env_dim - 1, env_dim // 2)


@dataclass
class RunSpec:
    """A single experiment run, fully reproducible from this dict."""

    family: str               # e.g. "e1_main", "e3_rollout"
    algo: str
    map_type: str
    env_dim: int
    seed: int
    total_budget: int = DEFAULT_TOTAL_BUDGET
    per_sim_budget: int = DEFAULT_PER_SIM_BUDGET
    sims_per_child: int = DEFAULT_SIMS_PER_CHILD
    n_checkpoints: int = 0    # 0 means single-goal (legacy behavior)
    start: tuple[int, int] | None = None
    goal: tuple[int, int] | None = None
    overrides: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.start is None or self.goal is None:
            s, g = default_start_goal(self.env_dim)
            self.start = self.start or s
            self.goal = self.goal or g

    @property
    def run_id(self) -> str:
        """Stable filesystem-friendly identifier."""
        ov = "_".join(f"{k}-{v}" for k, v in sorted(self.overrides.items())) or "base"
        return (f"{self.algo}__{self.map_type}__d{self.env_dim}__"
                f"cp{self.n_checkpoints}__b{self.total_budget}__"
                f"s{self.seed}__{ov}")

    def to_dict(self) -> dict:
        return asdict(self)
