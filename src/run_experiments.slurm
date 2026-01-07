#!/bin/bash
#SBATCH --job-name=damn
#SBATCH --output=logs/job.out
#SBATCH --error=logs/job.err
#SBATCH --cpus-per-task=27
#SBATCH --mem=100G
#SBATCH --ntasks=1
#SBATCH --partition=members
#SBATCH --nodelist=ant16


echo "Starting mcts job with 27 workers ..."
source ../venv/bin/activate
python run_experiments.py
tar -czvf log.tar.gz log/