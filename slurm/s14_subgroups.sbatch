#!/usr/bin/env bash
# Archaea vs bacteria: is the pocket exclusion one prokaryotic result, or one group's result?
# Reads the descriptor table stages 10-11 already wrote -- no structures, no embeddings, minutes not hours.
#   AlphaFold set: sbatch slurm/s14_subgroups.sbatch
#   ESMFold set:   sbatch --export=ALL,TAG=_esmfold slurm/s14_subgroups.sbatch
#   by phylum:     sbatch --export=ALL,SPLIT=phylum,TOP=4 slurm/s14_subgroups.sbatch
#SBATCH --job-name=serprot_subgroups
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"
echo "TAG=${TAG:-<none>}  SPLIT=${SPLIT:-domain}"
python scripts/19_subgroups.py \
    --config "${CONFIG:-config.yaml}" \
    --radius "${RADIUS:-10}" \
    --split "${SPLIT:-domain}" \
    --top "${TOP:-2}" \
    --min-clades "${MIN_CLADES:-25}" \
    --n-boot "${N_BOOT:-2000}" \
    --n-perm "${N_PERM:-2000}" \
    ${DESCRIPTORS:+--descriptors "$DESCRIPTORS"} \
    ${TAG:+--tag "$TAG"}
