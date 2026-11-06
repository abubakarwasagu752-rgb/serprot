#!/usr/bin/env bash
#SBATCH --job-name=serprot_cluster
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"
python scripts/02_build_labels.py  --config "${CONFIG:-config.yaml}"
python scripts/03_cluster_split.py --config "${CONFIG:-config.yaml}" --threads "${SLURM_CPUS_PER_TASK:-8}"
