#!/usr/bin/env bash
# Download AlphaFold models. Submit as an array to shard the work, e.g. 10 shards:
#   sbatch --array=0-9 slurm/s05_structures.sbatch
#SBATCH --job-name=serprot_struct
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"
python scripts/10_fetch_structures.py --config "${CONFIG:-config.yaml}" \
    --workers "${WORKERS:-8}" --shard "${SLURM_ARRAY_TASK_ID:-0}" --n-shards "${SLURM_ARRAY_TASK_COUNT:-1}"
