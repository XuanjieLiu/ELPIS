#!/bin/bash
# Submit from the login node; every Python command runs in the inner srun.
set -euo pipefail
export ELPIS_VISION_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
salloc -p ws-ia -N 1 --ntasks=1 --cpus-per-task=4 \
    --gres=gpu:nvidia-rtx-5000-ada-generation:1 --mem=32G --time=01:00:00 \
    srun bash -lc '
        set -euo pipefail
        source "${ELPIS_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
        conda activate "${ELPIS_CONDA_ENV:-xuanjie}"
        cd "$ELPIS_VISION_ROOT"
        export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
        export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
        while IFS= read -r run; do
            python -u VQ/batch_train.py "$run" --smoke-test
        done < reproduction/smoke_runs.txt
    '
