#!/bin/bash
#SBATCH --job-name=please_work
#SBATCH --output=logs/job.out
#SBATCH --error=logs/job.err
#SBATCH --cpus-per-task=25
#SBATCH --mem=200G
#SBATCH --ntasks=1
#SBATCH --partition=members


echo "Starting mcts job with 25 workers ..."
source ../venv/bin/activate
python run_experiments.py
