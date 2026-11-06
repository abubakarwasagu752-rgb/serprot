#!/usr/bin/env bash
# Joint distance x burial stratification: is the gradient about the active site, or about exposure?
#   AlphaFold set: sbatch slurm/s11_burial.sbatch
#   ESMFold set:   sbatch --export=ALL,TAG=_esmfold,STRUCT_DIR=data/structures_esmfold \
#                      slurm/s11_burial.sbatch
#SBATCH --job-name=serprot_burial
#SBATCH --partition=gpul40,gpu4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"
# diagnostic only -- never let it kill the job (env.sh sets -e, and a wedged GPU makes this fail)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo 'WARNING: nvidia-smi failed'
echo "TAG=${TAG:-<none>}  STRUCT_DIR=${STRUCT_DIR:-<default>}"
python scripts/16_burial_distance.py \
    --config "${CONFIG:-config.yaml}" \
    --n-boot "${N_BOOT:-800}" \
    --n-perm "${N_PERM:-800}" \
    --batch "${BATCH:-8}" \
    --n-burial "${N_BURIAL:-3}" \
    ${DIST_EDGES:+--dist-edges "$DIST_EDGES"} \
    ${STRUCT_DIR:+--structures-dir "$STRUCT_DIR"} \
    ${TAG:+--tag "$TAG"}
