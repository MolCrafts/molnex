#!/bin/bash
# NaN-term probe (diagnostic). Not a train.

set -euo pipefail

echo "=== node $(hostname) arch $(uname -m) $(date -Iseconds) ==="
source "${LMOD_PKG:-/software/sse2/el9_epyc9005/manual/lmod/lmod/lmod}/init/bash"
module purge
module load GPU/buildenv-gcccuda/2026.03-cu13.0

MOLCRAFTS="${MOLCRAFTS:-/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts}"
MOLNEX="${MOLCRAFTS}/molnex"
PY="${PY:-/home/jicli594/work/.aarch64/bin/python}"
export RUSTUP_HOME="${MOLNEX}/.rustup-gh200"
export CARGO_HOME="${MOLNEX}/.cargo-gh200"
export CARGO_TARGET_DIR="${MOLCRAFTS}/molrs/molrs-python/target-aarch64"
export PATH="${CARGO_HOME}/bin:${PATH}"
GCC_LIBDIR="$(dirname "$(g++ -print-file-name=libstdc++.so)")"
export LD_LIBRARY_PATH="${GCC_LIBDIR}:${LD_LIBRARY_PATH:-}"
MOLPY_SRC="${MOLCRAFTS}/.worktrees/molpy-0.13.1/src"
export PYTHONPATH="${MOLNEX}:${MOLNEX}/src:${MOLCRAFTS}/molhub/src:${MOLPY_SRC}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "${MOLNEX}"

WHEEL_DIR="${MOLNEX}/.wheels-gh200"
WHEEL="$(ls -1t "${WHEEL_DIR}"/molcrafts_molrs-*.whl 2>/dev/null | head -1 || true)"
if [ -z "${WHEEL}" ]; then
  (cd "${MOLCRAFTS}/molrs/molrs-python" && "${PY}" -m maturin build --release --out "${WHEEL_DIR}")
  WHEEL="$(ls -1t "${WHEEL_DIR}"/molcrafts_molrs-*.whl | head -1)"
fi
"${PY}" -m pip install --force-reinstall --no-deps "${WHEEL}"

"${PY}" scripts/mm_param_learning/_probe_nan_term.py

echo "=== DONE $(date -Iseconds) ==="
