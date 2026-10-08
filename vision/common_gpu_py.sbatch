#!/bin/bash
#SBATCH --partition=ws-ia
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia-rtx-5000-ada-generation:1
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --job-name=elpis-vision
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail
if (( $# < 1 )); then
    echo "Usage: sbatch common_gpu_py.sbatch VQ/batch_train.py 'run_name' [--smoke-test]" >&2
    exit 2
fi

# Submit from the vision directory (or set ELPIS_VISION_ROOT explicitly).
export ELPIS_VISION_ROOT="${ELPIS_VISION_ROOT:-${SLURM_SUBMIT_DIR}}"
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
srun bash -lc '
    set -euo pipefail
    source "${ELPIS_CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
    conda activate "${ELPIS_CONDA_ENV:-xuanjie}"
    cd "$ELPIS_VISION_ROOT"
    exec python -u "$@"
' elpis "$@"
