#!/bin/bash

###############################################
# Self-contained Slurm Job Array Parameter Sweep
# Run this on the login node:
#
#   ./run_grid_array.sh
#
###############################################

# --- CONFIGURE YOUR PARTITION/RESOURCES HERE ---
PARTITION="members"
CPUS=1

# --- PARAMETER SPACE ---
MAP_TYPES=("random_map")
ENV_DIMS=(30)
TOTAL_BUDGETS=(10000 100000 500000 1000000 5000000)
PER_SIM_BUDGETS=(20 30 50 100)
NUM_SIMS_LIST=(100 250 500)
TREE_METHODS=(0 1 2)
ROOT_METHODS=(0 1 2)
ROLLOUT_METHODS=(0 1)
SEEDS=(42 420 1337 2024 7777 9999 12345 54321 88888 99999)
ARCHIVES=(20)

START_X=0
START_Y=0
GOAL_X=29
GOAL_Y=29

# Output file for combinations
PARAM_FILE="param_grid.csv"

echo "Generating parameter grid: $PARAM_FILE"
echo "map,env_dim,budget,per_sim,num_sims,tree_sel,root_sel,rollout,seed,archive,start_x,start_y,goal_x,goal_y" > $PARAM_FILE

# Generate Cartesian product
for MAP in "${MAP_TYPES[@]}"; do
for ENV_DIM in "${ENV_DIMS[@]}"; do
for BUDGET in "${TOTAL_BUDGETS[@]}"; do
for PER_SIM in "${PER_SIM_BUDGETS[@]}"; do
for NUM_SIMS in "${NUM_SIMS_LIST[@]}"; do
for TREESEL in "${TREE_METHODS[@]}"; do
for ROOTSEL in "${ROOT_METHODS[@]}"; do
for ROLLOUT in "${ROLLOUT_METHODS[@]}"; do
for SEED in "${SEEDS[@]}"; do
for ARCHIVE in "${ARCHIVES[@]}"; do

    echo "$MAP,$ENV_DIM,$BUDGET,$PER_SIM,$NUM_SIMS,$TREESEL,$ROOTSEL,$ROLLOUT,$SEED,$ARCHIVE,$START_X,$START_Y,$GOAL_X,$GOAL_Y" >> $PARAM_FILE

done
done
done
done
done
done
done
done
done
done

# Count lines (minus header)
NUM_LINES=$(($(wc -l < $PARAM_FILE) - 1))
LAST_INDEX=$((NUM_LINES - 1))

echo "Generated $NUM_LINES parameter combinations."
echo "Submitting as Slurm array (0-$LAST_INDEX)"

###############################################
# Create a Slurm array file
###############################################
SLURM_FILE="run_array.slurm"

cat <<EOF > $SLURM_FILE
#!/bin/bash
#SBATCH --job-name=mcts_array
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.err
#SBATCH --partition=$PARTITION
#SBATCH --cpus-per-task=$CPUS
#SBATCH --array=0-$LAST_INDEX

LINE=\$(awk "NR==(\$SLURM_ARRAY_TASK_ID + 2)" $PARAM_FILE)

IFS=',' read MAP ENV_DIM BUDGET PER_SIM NUM_SIMS TREESEL ROOTSEL ROLLOUT SEED ARCHIVE START_X START_Y GOAL_X GOAL_Y <<< "\$LINE"

echo "Running task \$SLURM_ARRAY_TASK_ID on node \$HOSTNAME"
echo "Parameters:"
echo "\$LINE"

python3 main_cluster.py \
    --map "\$MAP" \
    --env_dim "\$ENV_DIM" \
    --start_x "\$START_X" \
    --start_y "\$START_Y" \
    --goal_x "\$GOAL_X" \
    --goal_y "\$GOAL_Y" \
    --budget "\$BUDGET" \
    --per_sim_budget "\$PER_SIM" \
    --num_sims "\$NUM_SIMS" \
    --rollout_method "\$ROLLOUT" \
    --root_sel "\$ROOTSEL" \
    --tree_sel "\$TREESEL" \
    --max_archive "\$ARCHIVE" \
    --seed "\$SEED"

EOF

mkdir -p logs

# Submit job array
sbatch "$SLURM_FILE"

echo "Submitted Slurm job array."
