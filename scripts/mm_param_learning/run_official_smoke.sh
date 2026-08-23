#!/bin/bash
set -euo pipefail
echo "=== node $(hostname) arch $(uname -m) $(date -Iseconds) ==="
echo "=== SLURM_JOB_ID=${SLURM_JOB_ID:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} ==="

source "${LMOD_PKG:-/software/sse2/el9_epyc9005/manual/lmod/lmod/lmod}/init/bash"
module purge
module load GPU/buildenv-gcccuda/2026.03-cu13.0 || true

MOLCRAFTS=/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts
MOLNEX="${MOLCRAFTS}/molnex"
PY="${PY:-/home/jicli594/work/.aarch64/bin/python}"
export MOLHUB_HOME="${MOLHUB_HOME:-${MOLCRAFTS}/workspaces/molhub-home}"
export RUSTUP_HOME="${MOLNEX}/.rustup-gh200"
export CARGO_HOME="${MOLNEX}/.cargo-gh200"
export PATH="${CARGO_HOME}/bin:${PATH}"
GCC_LIBDIR="$(dirname "$(g++ -print-file-name=libstdc++.so)" 2>/dev/null || echo /usr/lib64)"
export LD_LIBRARY_PATH="${GCC_LIBDIR}:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${MOLNEX}:${MOLNEX}/src:${MOLCRAFTS}/molhub/src:${MOLCRAFTS}/molpy/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "${MOLNEX}"
echo "=== python ${PY} MOLHUB_HOME=${MOLHUB_HOME} ==="
"${PY}" -c "import platform, torch; print('arch', platform.machine(), 'torch', torch.__version__, 'cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

echo "=== official OMol25 smoke (load + 16 train steps) ==="
exec "${PY}" scripts/mm_param_learning/prod_day.py smoke
