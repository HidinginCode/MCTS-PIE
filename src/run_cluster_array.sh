#!/bin/bash

###############################################
# MCTS Batch-Based Parameter Sweep for SLURM
# Avoids Slurm Array Limits (MaxArraySize=1001)
###############################################

PARTITION="members"
CPUS=1
BATCH_SIZE=500   # Number of parameter lines per job

###############################################
# ORIGINAL PARAMETERS
###############################################
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

###############################################
# Generate CSV
###############################################
PARAM_FILE="$(pwd)/param_grid.csv"

echo "Generating parameter grid: $PARAM_FILE"
echo "map,env_dim,budget,per_sim,num_sims,tree_sel,root_sel,rollout,seed,archive,start_x,start_y,goal_x,goal_y" > "$PARAM_FILE"

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

    echo "$MAP,$ENV_DIM,$BUDGET,$PER_SIM,$NUM_SIMS,$TREESEL,$ROOTSEL,$ROLLOUT,$SEED,$ARCHIVE,$START_X,$START_Y,$GOAL_X,$GOAL_Y" >> "$PARAM_FILE"

done; done; done; done; done; done; done; done; done; done

NUM_LINES=$(($(wc -l < "$PARAM_FILE") - 1))
echo "Generated $NUM_LINES parameter combinations."

###############################################
# Prepare batches
###############################################
NUM_BATCHES=$(( (NUM_LINES + BATCH_SIZE - 1) / BATCH_SIZE ))

echo "Submitting $NUM_BATCHES batches (batch size = $BATCH_SIZE)"

mkdir -p logs

for (( BATCH=0; BATCH < NUM_BATCHES; BATCH++ )); do

    START=$(( BATCH * BATCH_SIZE + 1 ))   # +1 offset for data rows, not header
    END=$(( START + BATCH_SIZE - 1 ))

    if (( END > NUM_LINES )); then
        END=$NUM_LINES
    fi

    SLURM_FILE="run_batch_${BATCH}.slurm"

cat <<EOF > "$SLURM_FILE"
#!/bin/bash
#SBATCH --job-name=mcts_batch_$BATCH
#SBATCH --output=logs/batch_${BATCH}.out
#SBATCH --error=logs/batch_${BATCH}.err
#SBATCH --partition=$PARTITION
#SBATCH --cpus-per-task=$CPUS

cd \$SLURM_SUBMIT_DIR

echo "Activating venv"
source ../venv/bin/activate

echo "Running batch $BATCH: rows $START to $END"

for LINE_NUM in \$(seq $START $END); do

    LINE=\$(awk "NR==(\$LINE_NUM + 1)" "$PARAM_FILE")

    IFS=',' read MAP ENV_DIM BUDGET PER_SIM NUM_SIMS TREESEL ROOTSEL ROLLOUT SEED ARCHIVE START_X START_Y GOAL_X GOAL_Y <<< "\$LINE"

    echo "Processing row \$LINE_NUM : \$LINE"

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

done
EOF

    echo "Submitting batch $BATCH (rows $START-$END)"
    sbatch "$SLURM_FILE"

done

echo "All batches submitted."