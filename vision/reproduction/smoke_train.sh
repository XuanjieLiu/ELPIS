#!/bin/bash
# Run all dataset representatives in the active Python environment.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export MPLBACKEND=Agg
while IFS= read -r run; do
    python -u VQ/batch_train.py "$run" --smoke-test
done < reproduction/smoke_runs.txt
