#!/usr/bin/env bash
# Submit the whole pipeline with dependencies.
#   PARTITION=gpul40 CPU_PARTITION=cpu bash slurm/run_all.sh     (add ACCOUNT=... if your cluster needs one)
#   Default trains only joint + disent; ablations: VARIANTS='joint plain disent noadv nodcor'
# Optional: CPU_PARTITION=cpu (data stage), CONFIG=config.yaml, EXTRA="--qos=normal"
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
GPU_FLAGS="${PARTITION:+--partition=$PARTITION} ${ACCOUNT:+--account=$ACCOUNT} ${EXTRA:-}"
CPU_FLAGS="${CPU_PARTITION:+--partition=$CPU_PARTITION} ${ACCOUNT:+--account=$ACCOUNT} ${EXTRA:-}"
export CONFIG="${CONFIG:-config.yaml}"

j1=$(sbatch --parsable $CPU_FLAGS slurm/s01_data.sbatch)
echo "data:       $j1"
j2=$(sbatch --parsable $GPU_FLAGS --dependency=afterok:$j1 slurm/s02_baselines.sbatch)
echo "baselines:  $j2"

deps="$j2"
for v in ${VARIANTS:-joint disent}; do   # ablations: VARIANTS='joint plain disent noadv nodcor'
  j=$(sbatch --parsable $GPU_FLAGS --dependency=afterok:$j1 --array=0-4 --export=ALL,VARIANT=$v,SPLIT=cv slurm/s03_train.sbatch)
  echo "train $v cv:  $j"; deps="$deps:$j"
done
for v in joint disent; do
  j=$(sbatch --parsable $GPU_FLAGS --dependency=afterok:$j1 --array=0-2 --export=ALL,VARIANT=$v,SPLIT=lofo slurm/s03_train.sbatch)
  echo "train $v lofo: $j"; deps="$deps:$j"
done
ja=$(sbatch --parsable $GPU_FLAGS --dependency=afterok:$deps slurm/s04_analyze.sbatch)
echo "analysis:   $ja"
echo "monitor with: squeue -u \$USER   |   logs in ./logs"
