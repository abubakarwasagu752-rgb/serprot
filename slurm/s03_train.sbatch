#!/usr/bin/env bash
# Job array over folds. Submit with:  sbatch --array=0-4 --export=ALL,VARIANT=disent,SPLIT=cv slurm/s03_train.sbatch
# (for SPLIT=lofo use --array=0-2: one task per held-out family)
#SBATCH --job-name=serprot_train
#SBATCH --partition=gpul40
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
source "${SLURM_SUBMIT_DIR:-.}/slurm/env.sh"
python scripts/05_train_disentangled.py --config "${CONFIG:-config.yaml}" \
    --variant "${VARIANT:?set VARIANT}" --split "${SPLIT:-cv}" --fold "${SLURM_ARRAY_TASK_ID:?run as an array job}"
