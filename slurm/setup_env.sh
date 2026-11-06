#!/usr/bin/env bash
# One-time environment setup. Run on the LOGIN node (installs only; no jobs run here).
#   bash slurm/setup_env.sh
set -euo pipefail
ENV="${SERPROT_ENV:-serprot}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n "$ENV" python=3.11
conda activate "$ENV"
conda install -y -c conda-forge -c bioconda mmseqs2
# pip wheels of torch bundle their own CUDA runtime; only the NVIDIA driver must be recent enough
pip install -r requirements.txt
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__)"
which mmseqs && mmseqs version
echo "env '$ENV' ready. Next: sbatch slurm/smoke_gpu.sbatch"
