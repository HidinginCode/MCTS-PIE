"""Shared experiment harness for the MCTS-PIE paper.

Modules:
    config        - RunSpec dataclass, shared seed list, 10-map suite.
    provenance    - git/pip/map snapshot for reproducibility.
    metrics       - HV / IGD+ / epsilon-indicator / normalization.
    stats         - Friedman, Wilcoxon-Holm, Vargha-Delaney, bootstrap CI, CD diagram.
    runner        - Generic resumable parallel runner over RunSpec lists.
"""
