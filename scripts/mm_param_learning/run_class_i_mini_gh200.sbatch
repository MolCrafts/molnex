#!/bin/bash
# Bounded Class-I mini train: OMol25-small 8:2 self-split, 12 epochs.
# Descent + step-time check. Not a production train.
#
#   molq submit slurm --name class-i-mini --partition gpu \
#     --account naiss2026-1-53-gpu --gpus 1 --cpus 8 --time 1h \
#     --workdir /nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts/molnex \
#     bash scripts/mm_param_learning/run_class_i_mini_gh200.sbatch

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
export PYTHONPATH="${MOLNEX}:${MOLCRAFTS}/molhub/src:${MOLCRAFTS}/molexp/src:${MOLCRAFTS}/molcfg/src:${MOLCRAFTS}/molpy/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "${MOLNEX}"

echo "=== python ${PY} ==="
"${PY}" -c "import platform, torch; print('arch', platform.machine(), 'torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

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
"${PY}" -c "import lmdb" 2>/dev/null || "${PY}" -m pip install -q lmdb

echo "=== ClassIFit mini: 256 train / 64 val / 12 epochs (not molexp declared-run replay) ==="
"${PY}" - <<'PY'
from scripts.mm_param_learning.class_i_fit import ClassIFit, load_omol25_split

train, val = load_omol25_split(role="val-neutral", n_keep=256, max_atoms=32)
print("n_train", len(train), "n_val", len(val), flush=True)
metrics = ClassIFit(
    train,
    val,
    hidden=64,
    n_blocks=4,
    max_atoms=16384,
    epochs=12,
    device="cuda",
    eval_every_epochs=2,
    energy_scope="full",
).run()
print(metrics, flush=True)
PY

echo "=== DONE $(date -Iseconds) ==="
