# MCTS-PIE
![Python](https://img.shields.io/badge/python-3.13.1-blue)
## Introduction
This project contains a path influenced ennvironment problem, and a monte carlo tree search approach to solve it.

## Requirements
- Dependencies are listed in [requirements.txt](requirements.txt)

## Classes
### Map
The map class is the basic construct of the problem.
It is always a square map that has its dimension set on creation.
It is modeled using 2 dimensional lists, for reasons of comuting speed and contains random values between `0` and `1` in each cell.

### Agent
The agent class is used to simulate the agent in the environment and collect metrics.
Each agent has an identifier attribute that is its `ID` on creation.
Metrics collected are `step_count`, `amount_of_shifts` and `energy_consumption`.
The agent also holds the `path` that it takes to traverse the map.