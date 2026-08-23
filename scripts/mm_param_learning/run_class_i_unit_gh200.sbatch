#!/bin/bash
# Targeted unit tests for the Class-I forcefield + head changes. Not a train.

set -euo pipefail

echo "=== node $(hostname) arch $(uname -m) $(date -Iseconds) ==="
echo "=== SLURM_JOB_ID=${SLURM_JOB_ID:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} ==="

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
if [ ! -d "${MOLPY_SRC}" ]; then
  echo "=== create molpy-0.13.1 worktree (molpy dev checkout is stale at 0.7.0) ==="
  git -C "${MOLCRAFTS}/molpy" worktree add --detach "${MOLCRAFTS}/.worktrees/molpy-0.13.1" master
fi
export PYTHONPATH="${MOLNEX}:${MOLNEX}/src:${MOLCRAFTS}/molhub/src:${MOLPY_SRC}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "${MOLNEX}"

echo "=== python ${PY} ==="
"${PY}" -c "import platform, torch; print('arch', platform.machine(), 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"

WHEEL_DIR="${MOLNEX}/.wheels-gh200"
mkdir -p "${WHEEL_DIR}"
WHEEL="$(ls -1t "${WHEEL_DIR}"/molcrafts_molrs-*.whl 2>/dev/null | head -1 || true)"
if [ -z "${WHEEL}" ]; then
  echo "=== compile current molrs into .aarch64 ==="
  (
    cd "${MOLCRAFTS}/molrs/molrs-python"
    "${PY}" -m maturin build --release --out "${WHEEL_DIR}"
  )
  WHEEL="$(ls -1t "${WHEEL_DIR}"/molcrafts_molrs-*.whl | head -1)"
else
  echo "=== reuse ${WHEEL} ==="
fi
"${PY}" -m pip install --force-reinstall --no-deps "${WHEEL}"

echo "=== import smoke (molpy from worktree, molrs from wheel) ==="
"${PY}" -c "import molpy, molrs, molix; print('molpy', molpy.__version__ if hasattr(molpy,'__version__') else 'n/a', 'ok')"

"${PY}" -m pytest -q \
  tests/test_molpot/test_forcefield/test_potentials.py \
  tests/test_molpot/test_forcefield/test_analytic_forces.py \
  tests/test_molpot/test_forcefield/test_coulomb.py \
  tests/test_molpot/test_composition/test_byteff_heads.py \
  tests/test_molpot/test_composition/test_mm_heads.py \
  tests/test_molpot/test_charge/test_eem.py \
  tests/test_scripts/test_mm_param_learning/test_class_i_fit.py

echo "=== DONE $(date -Iseconds) ==="
