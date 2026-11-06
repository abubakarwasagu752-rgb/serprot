#!/usr/bin/env bash
# Refold proteins with ESMFold. Shard across GPUs with an array:
#   sbatch --array=0-3 slurm/s12_esmfold.sbatch
# Smoke test first (one short job):  LIMIT=20 sbatch --array=0 slurm/s12_esmfold.sbatch
#SBATCH --job-name=serprot_esmfold
#SBATCH --partition=gpul40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
# gpul40 ONLY, deliberately: the gpu4090 nodes have 40 GB host RAM (too little for a 48G request)
# and 24 GB of GPU memory, which ESMFold can exceed on a ~700-residue chain. The L40s have 128 GB
# host and 46 GB GPU. 48G / 8h / 8 cpus matches the GPU jobs that already scheduled here.
source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"
# diagnostic only -- never let it kill the job (env.sh sets -e, and a wedged GPU makes this fail)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo 'WARNING: nvidia-smi failed; the GPU on this node may be unusable'
python scripts/17_esmfold.py --config "${CONFIG:-config.yaml}" \
    --shard "${SLURM_ARRAY_TASK_ID:-0}" --n-shards "${SLURM_ARRAY_TASK_COUNT:-1}" \
    --budget "${BUDGET:-8000}" --max-len "${MAX_LEN:-700}" \
    --min-len "${MIN_LEN:-0}" --chunk-size "${CHUNK:-64}" ${LIMIT:+--limit "$LIMIT"}
