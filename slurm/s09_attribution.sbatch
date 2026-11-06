#!/usr/bin/env bash
# Exact per-residue attribution: where does ESM-2 read temperature from?
# Needs a GPU (one forward pass per protein through ESM-2 650M).
#SBATCH --job-name=serprot_attrib
#SBATCH --partition=gpul40,gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/14_residue_attribution.py --config "${CONFIG:-config.yaml}" \
    --radius "${RADIUS:-10}" --n-controls "${N_CONTROLS:-3}" \
    --n-boot "${N_BOOT:-2000}" --n-perm "${N_PERM:-2000}" --batch "${BATCH:-8}"
