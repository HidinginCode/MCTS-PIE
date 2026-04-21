# MCTS-PIE
![Python](https://img.shields.io/badge/python-3.13-blue)

## Introduction
MCTS-PIE is a research project implementing a **multi-objective Monte Carlo Tree Search (MCTS)** for a **Path-Influenced Environment (PIE)** problem.

An agent navigates a weighted grid map from a start position to a waypoint goal and back. At each step the agent also shifts the weight of the cell it enters to an adjacent cell, modifying the map. The algorithm simultaneously optimises three competing objectives:

- **Step count** — minimise the total number of moves
- **Weight shifted** — minimise the total weight displaced during traversal
- **Distance to goal** — minimise remaining distance

Four map types of varying difficulty are supported: `easy_map`, `checkerboard_map`, `random_map`, and `meandering_river_map`, each available in 20×20, 35×35, and 50×50 grids.

An A\* epsilon-constraint search serves as a reference solver.

## Installation

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run a quick local experiment from the `src/` directory:

```bash
cd src
python main.py
```

To run a parametrised simulation programmatically, call `simulations()` in [src/main.py](src/main.py):

```python
simulations(
    map="easy_map",
    env_dim=50,
    start=(0, 25),
    goal=(49, 25),
    budget=300000,
    per_sim_budget=75,
    number_of_sims=50,
    rollout_method=1,       # 0=light, 1=square_sampling, 2=distance_weight
    root_selection_method=0,
    tree_selection_method=0, # 0=UCB, 1=HV, 2=CD, 3=AEGA
    seed=420,
)
```

Results are written to `src/log/` as pickle files and can be analysed with the `Analyzer` class.

## Requirements
Dependencies are listed in [requirements.txt](requirements.txt). Install with `pip install -r requirements.txt`.

## Project Structure

```
src/
├── main.py          # Entry point and experiment wrappers
├── environment.py   # Grid map generation and storage
├── controller.py    # Agent movement and weight-shifting logic
├── node.py          # MCTS tree node
├── mc_tree.py       # MCTS tree (tree policy, rollouts, backpropagation)
├── helper.py        # Pareto utilities, hypervolume, crowding distance
├── astar.py         # A* epsilon-constraint reference solver
├── analyzer.py      # Result loading, plotting, statistical analysis
├── logger.py        # Run logging to disk
├── maps/            # Serialised map files (auto-generated)
├── log/             # MCTS result logs
└── a_star_log/      # A* result logs
```

## Modules

### Environment
Represents the weighted obstacle grid. Maps are generated once and cached as pickle files under `src/maps/`. Supported types:

| Map type | Description |
|---|---|
| `random_map` | Uniform-random cell weights |
| `checkerboard_map` | Sinusoidal weight pattern |
| `easy_map` | Random weights with a cleared corridor to the goal |
| `meandering_river_map` | Gaussian-smoothed S-shaped low-weight river |

### Controller
Executes agent movement and weight-shifting on the environment. Each `move(move_dir, shift_dir)` call advances the agent one step and displaces the weight of the entered cell to an adjacent cell. Tracks `step_count`, `weight_shifted`, and `distance_to_goal`.

### Node
A node in the MCTS tree. Stores the `Controller` state, visit count, averaged objective values, last move, and a Pareto archive of promising paths seen below this node.

### MctsTree
Implements the full MCTS loop:

- **Tree policy** — progressive widening + configurable child-selection strategy
- **Child selection** strategies:
  - `UCB` (0) — multi-objective UCB1 with Pareto dominance
  - `HV` (1) — hypervolume-weighted archive sampling
  - `CD` (2) — crowding-distance-weighted archive sampling
  - `AEGA` (3) — adaptive epsilon-grid archiving selection
- **Rollout** strategies:
  - `light` (0) — uniform random moves
  - `square_sampling` (1) — greedy moves towards sampled local waypoints
  - `distance_weight` (2) — greedy moves preferring low-weight, goal-approaching cells
- **Backpropagation** — running mean update + per-node Pareto archive maintenance
- **Root progression** — hypervolume-based best-child root advancement with sibling pruning

### Helper
Static utility methods: z-score normalisation, min-max normalisation, Pareto front extraction, hypervolume contributions (via pymoo), crowding distance, epsilon clustering, and adaptive epsilon archiving.

### A_Star
Epsilon-constraint A\* reference solver. Phase 1 computes the minimum-step path; phase 2 sweeps an epsilon budget to enumerate the Pareto front of (step count, weight shifted) trade-offs. Includes shifting optimisation to minimise weight displacement along fixed paths.

### Analyzer
Loads pickle logs from `src/log/`, aggregates solutions across runs, plots Pareto fronts per configuration, and performs statistical significance testing across algorithm configurations.

### Logger
Creates structured log directories and serialises solution data to disk after each MCTS run.
