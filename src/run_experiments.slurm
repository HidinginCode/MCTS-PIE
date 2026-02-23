#!/bin/bash
#SBATCH --job-name=experiments
#SBATCH --partition=members
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=110G
#SBATCH --array=0-11
#SBATCH --exclude=ant13

source ../venv/bin/activate
python run_experiments.py