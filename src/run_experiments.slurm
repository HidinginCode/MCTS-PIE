#!/bin/bash
#SBATCH --job-name=damn
#SBATCH --output=logs/job.out
#SBATCH --error=logs/job.err
#SBATCH --cpus-per-task=40
#SBATCH --mem=320G
#SBATCH --ntasks=1
#SBATCH --partition=members


echo "Starting mcts job with 40 workers ..."
source ../venv/bin/activate
python run_experiments.py
