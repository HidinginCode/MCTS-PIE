#!/bin/bash
#SBATCH --job-name=mcts100
#SBATCH --partition=members
#SBATCH --ntasks=100
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/mpi128.out
#SBATCH --error=logs/mpi128.err

cd $SLURM_SUBMIT_DIR
source ../venv/bin/activate

srun python3 run_mpi.py
