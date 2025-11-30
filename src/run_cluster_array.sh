#!/bin/bash

PARTITION="members"
CPUS=1
MEMORY="16G"              # Memory per batch job
BATCH_SIZE=2000           # Larger batch size → fewer total jobs

############################################################
# PARAMETER SPACE
############################################################

MAP_TYPES=("random_map" "easy_map" "checkerboard_map" "bubble_in_the_middle_map")
ENV_DIMS=(20 30 50)
TOTAL_BUDGETS=(2500000 5000000 10000000)
PER_SIM_BUDGETS=(30 50 100)
NUM_SIMS_LIST=(150 250 500)
TREE_METHODS=(0 1 2)
ROOT_METHODS=(0 1 2)
ROLLOUT_METHODS=(0 1 2)
SEEDS=(40121 793533 65039 619614 747470 959765 400 304113 189599 705854 44091 611977 945788 732612 638137 771653 647051 991531 893323 506735 966665 848983 244727 856757 81117 457783 899077 539863 301629 127256 747959)
ARCHIVES=(20)

############################################################
# GENERATE PARAMETER GRID CSV
############################################################

PARAM_FILE="$(pwd)/param_grid.csv"

echo "Generating parameter grid → $PARAM_FILE"
echo "map,env_dim,budget,per_sim,num_sims,tree_sel,root_sel,rollout,seed,archive,start_x,start_y,goal_x,goal_y" \
    > "$PARAM_FILE"

for MAP in "${MAP_TYPES[@]}"; do
for ENV_DIM in "${ENV_DIMS[@]}"; do

    START_X=$((ENV_DIM / 2))
    START_Y=0
    GOAL_X=$((ENV_DIM / 2))
    GOAL_Y=$((ENV_DIM - 1))

for BUDGET in "${TOTAL_BUDGETS[@]}"; do
for PER_SIM in "${PER_SIM_BUDGETS[@]}"; do
for NUM_SIMS in "${NUM_SIMS_LIST[@]}"; do
for TREESEL in "${TREE_METHODS[@]}"; do
for ROOTSEL in "${ROOT_METHODS[@]}"; do
for ROLLOUT in "${ROLLOUT_METHODS[@]}"; do
for SEED in "${SEEDS[@]}"; do
for ARCHIVE in "${ARCHIVES[@]}"; do

    echo "$MAP,$ENV_DIM,$BUDGET,$PER_SIM,$NUM_SIMS,$TREESEL,$ROOTSEL,$ROLLOUT,$SEED,$ARCHIVE,$START_X,$START_Y,$GOAL_X,$GOAL_Y" \
        >> "$PARAM_FILE"

done; done; done; done; done; done; done; done

done
done

NUM_LINES=$(($(wc -l < "$PARAM_FILE") - 1))
echo "Total parameter combinations: $NUM_LINES"

############################################################
# SPLIT INTO BATCHES
############################################################

NUM_BATCHES=$(( (NUM_LINES + BATCH_SIZE - 1) / BATCH_SIZE ))
echo "Creating $NUM_BATCHES batch jobs (batch size = $BATCH_SIZE)"

mkdir -p logs

############################################################
# SUBMIT EACH BATCH (NO CONCURRENCY LIMITER)
############################################################

for (( BATCH=0; BATCH < NUM_BATCHES; BATCH++ )); do

    START=$(( BATCH * BATCH_SIZE + 1 ))
    END=$(( START + BATCH_SIZE - 1 ))
    (( END > NUM_LINES )) && END=$NUM_LINES

    SLURM_FILE="run_batch_${BATCH}.slurm"

cat <<EOF > "$SLURM_FILE"
#!/bin/bash
#SBATCH --job-name=mcts_batch_${BATCH}
#SBATCH --output=logs/batch_${BATCH}.out
#SBATCH --error=logs/batch_${BATCH}.err
#SBATCH --partition=${PARTITION}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEMORY}

cd \$SLURM_SUBMIT_DIR

echo "Running batch $BATCH (rows $START to $END)"
. /opt/spack/main/env.sh
module load python
source ../venv/bin/activate

for LINE_NUM in \$(seq ${START} ${END}); do
    LINE=\$(awk "NR==(\$LINE_NUM + 1)" "${PARAM_FILE}")
    IFS=',' read MAP ENV_DIM BUDGET PER_SIM NUM_SIMS TREESEL ROOTSEL ROLLOUT SEED ARCHIVE SX SY GX GY <<< "\$LINE"

    echo "Processing: \$LINE"
    python3 main_cluster.py \
        --map "\$MAP" \
        --env_dim "\$ENV_DIM" \
        --start_x "\$SX" \
        --start_y "\$SY" \
        --goal_x "\$GX" \
        --goal_y "\$GY" \
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

    echo "Submitting batch $BATCH (rows: $START-$END)"
    sbatch "$SLURM_FILE"

done

echo "All batch jobs submitted."
