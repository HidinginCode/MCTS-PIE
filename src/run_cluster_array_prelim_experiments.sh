#!/bin/bash
#SBATCH --job-name=mcts_mp
#SBATCH --partition=members
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --output=logs/mp.out
#SBATCH --error=logs/mp.err

cd $SLURM_SUBMIT_DIR
source ../venv/bin/activate

python3 -u run_mp.py \
    --processes $SLURM_CPUS_PER_TASK \
    --total_memory "$SLURM_MEM_PER_NODE"
