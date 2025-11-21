# main_cluster.py

import argparse
from main import simulations

def parse_args():
    parser = argparse.ArgumentParser(description="Cluster MCTS simulations runner")

    parser.add_argument("--map", type=str, required=True)
    parser.add_argument("--env_dim", type=int, required=True)
    parser.add_argument("--start_x", type=int, required=True)
    parser.add_argument("--start_y", type=int, required=True)
    parser.add_argument("--goal_x", type=int, required=True)
    parser.add_argument("--goal_y", type=int, required=True)
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--per_sim_budget", type=int, required=True)
    parser.add_argument("--num_sims", type=int, required=True)
    parser.add_argument("--rollout_method", type=int, required=True)
    parser.add_argument("--root_sel", type=int, required=True)
    parser.add_argument("--tree_sel", type=int, required=True)
    parser.add_argument("--max_archive", type=int, default=20)
    parser.add_argument("--seed", type=int, default=420)

    return parser.parse_args()


def main():
    args = parse_args()

    simulations(
        map=args.map,
        env_dim=args.env_dim,
        start=(args.start_x, args.start_y),
        goal=(args.goal_x, args.goal_y),
        budget=args.budget,
        per_sim_budget=args.per_sim_budget,
        number_of_sims=args.num_sims,
        rollout_method=args.rollout_method,
        root_selection_method=args.root_sel,
        tree_selection_method=args.tree_sel,
        max_pareto_path_archive=args.max_archive,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
