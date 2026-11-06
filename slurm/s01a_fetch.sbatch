#!/usr/bin/env bash
#SBATCH --job-name=serprot_fetch
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"
python scripts/00_check_access.py --config "${CONFIG:-config.yaml}" || echo "access check reported problems (see above)"
python scripts/01_fetch_sequences.py --config "${CONFIG:-config.yaml}"
python scripts/02_build_labels.py    --config "${CONFIG:-config.yaml}"
