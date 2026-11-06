#!/usr/bin/env bash
# Pocket-vs-temperature analysis. Pure NumPy -- no GPU code, so this belongs on the CPU partition.
#   AlphaFold set:  for R in 6 8 10; do RADIUS=$R sbatch slurm/s07_pocket_analysis.sbatch; done
#   ESMFold set:    for R in 6 8 10; do RADIUS=$R TAG=_esmfold sbatch --export=ALL,RADIUS=$R,TAG=_esmfold \
#                       slurm/s07_pocket_analysis.sbatch; done
#SBATCH --job-name=serprot_pockana
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "RADIUS=${RADIUS:-10}  TAG=${TAG:-<none>}  CPUS=${SLURM_CPUS_PER_TASK:-?}"
python scripts/12_pocket_analysis.py \
    --config "${CONFIG:-config.yaml}" \
    --n-boot "${N_BOOT:-2000}" \
    --n-perm "${N_PERM:-2000}" \
    --radius "${RADIUS:-10}" \
    --workers "${SLURM_CPUS_PER_TASK:-16}" \
    ${TAG:+--tag "$TAG"}
