#!/usr/bin/env bash
# End-to-end pipeline. Pass extra config overrides through, e.g.:
#   ./scripts/run_all.sh dataset.name=cifar10
set -euo pipefail

CONFIG="${CONFIG:-configs/default.yaml}"
EXTRA=("$@")

cd "$(dirname "$0")/.."

run() { echo -e "\n=== $1 ==="; python "scripts/$1" --config "$CONFIG" "${EXTRA[@]}"; }

run 00_download_weights.py   # verifies checkpoints (no-op if all present)
run 01_extract_features.py   # extract + cache raw features
run 02_svd_analysis.py       # Part A: SVD / RankMe / Two-NN
run 03_svm_probe.py          # Part B: linear SVM probe
run 04_connect_analysis.py   # connect spectral rank <-> SVM margin

# Optional extensions (no-op unless enabled in the config / overrides):
run 05_viewpoint.py          # Part C: ModelNet40 viewpoint consistency
run 06_perturbation.py       # Part D: perturbation stability

echo -e "\nDone. Results under results/"
