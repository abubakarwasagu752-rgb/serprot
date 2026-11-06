#!/usr/bin/env bash
#SBATCH --job-name=serprot_base
#SBATCH --partition=gpul40
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"
python scripts/04_embed_baselines.py --config "${CONFIG:-config.yaml}"
