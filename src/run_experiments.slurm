#!/bin/bash
#SBATCH --job-name=masters-thesis
#SBATCH --output=logs/job.out
#SBATCH --error=logs/job.err
#SBATCH --cpus-per-task=120
#SBATCH --mem=120G
#SBATCH --ntasks=1
#SBATCH --partition=members
#SBATCH --nodelist=ant6


echo "Starting mcts job with 27 workers ..."
source ../venv/bin/activate
python run_experiments.py
tar -czvf log.tar.gz log/