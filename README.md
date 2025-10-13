# MCTS-PIE
![Python](https://img.shields.io/badge/python-3.13.1-blue)
## Introduction
This project contains a path influenced ennvironment problem, and a monte carlo tree search approach to solve it.

## Requirements
- Dependencies are listed in [requirements.txt](requirements.txt)

## Classes
### Map
The map class is the basic construct of the problem. 
Each map has an identifier `ID` that is set on creation.   
It is always a square map that has its dimension set on creation.  
It is modeled using 2 dimensional lists, for reasons of comuting speed and contains random values between `0` and `1` in each cell.  
Cells can be merged together which adds up their densities (values in the cells).  
The newly created obstacle then takes more energy to move again.

### Agent
The agent class is used to simulate the agent in the environment and collect metrics.  
Each agent has an identifier attribute that is its `ID` on creation.  
Metrics collected are `step_count`, `amount_of_shifts`, `energy_consumption`, `weight_shifted`.  
The agent also holds the `path` that it takes to traverse the map.

### Controller
The controller class simulates the agent on the map.  
It is responsible for the agents movements and the shifts on the map.  
On creation it deepcopys the given map and saves it as `map_copy`.  
The agent is saved under the variable `current_agent`.  
The agents current position is stored in the variable `current_agent_position`.  

### Directions
The directions enum is used for simplicity.  
It contains all 4 cardinal directions and their corresponding coordinates (`[1,0], [0,1], ...`).

### Node
The node class is used to build the mcts tree.  
It containts the current `state`, the `parent` node and the `children` (children as a `dictionary`).


### State
The state class contains the `state` of everything in a given node of the MCTS tree.  
It holds all `agent metrics` and its `position` on the current map.  
Later this will also contain an evaluation method that gives us the `fitness` of the state.