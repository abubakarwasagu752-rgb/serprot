#!/usr/bin/env bash
# Elastic-network flexibility: does the pocket resist thermal adaptation in its DYNAMICS too?
# Pure NumPy/SciPy -- one eigendecomposition per protein, so this wants CPU cores, not a GPU.
#   AlphaFold set: sbatch slurm/s13_flexibility.sbatch
#   ESMFold set:   sbatch --export=ALL,TAG=_esmfold,STRUCT_DIR=data/structures_esmfold \
#                      slurm/s13_flexibility.sbatch
#SBATCH --job-name=serprot_flex
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"
echo "TAG=${TAG:-<none>}  STRUCT_DIR=${STRUCT_DIR:-<default>}  CPUS=${SLURM_CPUS_PER_TASK:-?}"
# each worker already uses one core; let BLAS keep to one thread so they do not fight each other
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python scripts/18_flexibility.py \
    --config "${CONFIG:-config.yaml}" \
    --radius "${RADIUS:-10}" \
    --cutoff "${CUTOFF:-7.3}" \
    --n-boot "${N_BOOT:-2000}" \
    --n-perm "${N_PERM:-2000}" \
    --workers "${WORKERS:-0}" \
    ${DIST_EDGES:+--dist-edges "$DIST_EDGES"} \
    ${STRUCT_DIR:+--structures-dir "$STRUCT_DIR"} \
    ${TAG:+--tag "$TAG"}
