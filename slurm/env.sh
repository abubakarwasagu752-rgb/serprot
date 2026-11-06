#!/usr/bin/env bash
# Sourced by every job.
set -euo pipefail
export PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"   # sbatch copies scripts, so use the submit dir
cd "$PROJECT"
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$PROJECT/.hf_cache}"            # model weights cached inside the project (650M model ~2.6 GB)
export TOKENIZERS_PARALLELISM=false

# --- activate the conda env created by slurm/setup_env.sh (override name with SERPROT_ENV)
set +u
if ! command -v conda >/dev/null 2>&1; then
  for c in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge" /opt/conda /opt/miniconda3; do
    [ -f "$c/etc/profile.d/conda.sh" ] && source "$c/etc/profile.d/conda.sh" && break
  done
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi
if command -v conda >/dev/null 2>&1; then conda activate "${SERPROT_ENV:-serprot}"; fi
set -u
