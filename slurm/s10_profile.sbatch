#!/usr/bin/env bash
# Thermal signal as a continuous function of distance from the catalytic serine.
# Needs a GPU for the attribution readout (pass --no-attribution to run composition-only on CPU).
#SBATCH --job-name=serprot_profile
#SBATCH --partition=gpul40,gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/15_distance_profile.py --config "${CONFIG:-config.yaml}" \
    --n-boot "${N_BOOT:-1000}" --n-perm "${N_PERM:-1000}" --batch "${BATCH:-8}" \
    ${EDGES:+--edges "$EDGES"}
