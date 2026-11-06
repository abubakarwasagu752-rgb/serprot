#!/usr/bin/env bash
# Sequence-masking experiment: does the ESM-2 temperature signal avoid the pocket?
# This one GENUINELY needs a GPU -- it embeds ~4 x N sequences with ESM-2 650M.
#   sbatch slurm/s08_mask.sbatch              (or: sbatch --partition=gpu4090 slurm/s08_mask.sbatch)
#SBATCH --job-name=serprot_mask
#SBATCH --partition=gpul40,gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/13_mask_experiment.py --config "${CONFIG:-config.yaml}" \
    --radius "${RADIUS:-10}" --n-controls "${N_CONTROLS:-2}" \
    --n-boot "${N_BOOT:-2000}" --batch "${BATCH:-16}"
