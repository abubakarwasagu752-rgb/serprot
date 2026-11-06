#!/usr/bin/env bash
# End-to-end test on synthetic data with a tiny random ESM (CPU is fine, a few minutes).
# Live downloads (UniProt / AlphaFold / Hugging Face / mmseqs) are NOT exercised.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf tests/synth
C=tests/config_synth.yaml
python tests/make_synth_config.py
python scripts/make_synthetic.py --root tests/synth --clades 45 --members 10 --tradeoff 0.6
python scripts/make_tiny_esm.py --out tests/synth/tiny_esm
python scripts/02_build_labels.py  --config $C
python scripts/03_cluster_split.py --config $C
python scripts/04_embed_baselines.py --config $C
for v in joint disent; do
  for f in 0 1 2 3; do python scripts/05_train_disentangled.py --config $C --variant $v --split cv --fold $f; done
done
for f in 0 1 2; do python scripts/05_train_disentangled.py --config $C --variant disent --split lofo --fold $f --epochs 3; done
python scripts/06_analyze.py --config $C --n-boot 500
python scripts/07_attribution_pockets.py --config $C --variant disent --split cv --pocket-file tests/synth/data/raw/pockets_truth.tsv --no-download
python scripts/08_candidates.py --config $C --model disent --split cv --min-ogt-spread 3 --min-breadth-spread 0.0
echo "SMOKE TEST PASSED"
