#!/bin/bash
#SBATCH --job-name=mcts_array
#SBATCH --output=logs/out_%A_%a.txt
#SBATCH --error=logs/err_%A_%a.txt
#SBATCH --array=0-49

#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --ntasks=1

#SBATCH --partition=members
#SBATCH --exclude=ant7,ant8,ant9,ant10,ant11,ant12,ant13,ant14,ant15,ant16,ant19,ant20,ant21,ant22,ant23




source ../venv/bin/activate
echo "START Worker $SLURM_ARRAY_TASK_ID"

# Count lines
TOTAL_LINES=$(($(wc -l < params.csv) - 1))
CHUNK=$(( TOTAL_LINES / 50 + 1 ))

START=$(( SLURM_ARRAY_TASK_ID * CHUNK + 1 ))
END=$(( START + CHUNK ))

echo "Processing lines $START to $END"

tail -n +2 params.csv | sed -n "${START},${END}p" | \
while IFS=',' read map_type env_dim budget per_sim_budget num_sims \
                       tree_method root_method rollout_method seed archive
do
    echo "Running map=$map_type env_dim=$env_dim seed=$seed"

    # Compute goal coordinates
    gx=$((env_dim - 1))
    gy=$((env_dim - 1))

    python3 - <<EOF
from main import simulations

simulations(
    map="${map_type}",
    env_dim=${env_dim},
    start=(0, 0),
    goal=(${gx}, ${gy}),
    budget=${budget},
    per_sim_budget=${per_sim_budget},
    number_of_sims=${num_sims},
    rollout_method=${rollout_method},
    root_selection_method=${root_method},
    tree_selection_method=${tree_method},
    max_pareto_path_archive=${archive},
    seed=${seed}
)
EOF

done

echo "END Worker $SLURM_ARRAY_TASK_ID"
