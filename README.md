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