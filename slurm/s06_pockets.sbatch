#!/usr/bin/env bash
# Pocket descriptors from the downloaded models. Shard with an array if the set is large:
#   sbatch --array=0-7 slurm/s06_pockets.sbatch
#SBATCH --job-name=serprot_pocket
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"
python scripts/11_pocket_descriptors.py --config "${CONFIG:-config.yaml}" \
    --workers "${SLURM_CPUS_PER_TASK:-16}" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" --n-shards "${SLURM_ARRAY_TASK_COUNT:-1}" ${STRUCT_DIR:+--structures-dir "$STRUCT_DIR"} ${TAG:+--tag "$TAG"}
