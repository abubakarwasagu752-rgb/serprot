#!/usr/bin/env bash
#SBATCH --job-name=serprot_analyze
#SBATCH --partition=gpul40
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"
python scripts/06_analyze.py            --config "${CONFIG:-config.yaml}"
python scripts/07_attribution_pockets.py --config "${CONFIG:-config.yaml}" --variant disent --split cv ${POCKET_FILE:+--pocket-file "$POCKET_FILE"}
python scripts/08_candidates.py         --config "${CONFIG:-config.yaml}" --model disent --split cv
