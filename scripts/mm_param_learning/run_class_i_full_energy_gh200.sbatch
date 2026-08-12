#!/bin/bash
#SBATCH --job-name=class-i-full-e
#SBATCH --account=naiss2025-5-447-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=class-i-full-energy-%j.out

# Class-I full-energy mini → molexp run on GH200 (aarch64).
# Do NOT use login-node x86 cargo/maturin. Build molrs from source with
# aarch64 rustup (PyPI 0.13.0 wheel lacks Neighbors required by molpy 0.13).
#
#   sbatch scripts/mm_param_learning/run_class_i_full_energy_gh200.sbatch

set -euo pipefail

echo "=== node $(hostname) arch $(uname -m) $(date -Iseconds) ==="
echo "=== SLURM_JOB_ID=${SLURM_JOB_ID:-} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} ==="

source "${LMOD_PKG:-/software/sse2/el9_epyc9005/manual/lmod/lmod/lmod}/init/bash"
module purge
module load GPU/buildenv-gcccuda/2026.03-cu13.0
module load GPU/Python/3.13.5-bare-gcc-2025b-eb

MOLCRAFTS="${MOLCRAFTS:-/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts}"
MOLNEX="${MOLCRAFTS}/molnex"
VENV="${MOLNEX}/.venv-gh200"
# aarch64-only toolchains (never x86 login cargo)
export RUSTUP_HOME="${MOLNEX}/.rustup-gh200"
export CARGO_HOME="${MOLNEX}/.cargo-gh200"
# strip x86 cargo from PATH if present
export PATH="$(echo "${PATH}" | tr ':' '\n' | grep -v 'x86_64/cargo' | paste -sd: -)"

cd "${MOLNEX}"

if [ ! -x "${VENV}/bin/python" ]; then
  echo "=== creating GH200 venv ==="
  python -m venv "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install -q --upgrade pip

echo "=== torch + deps ==="
python -m pip install -q "torch>=2.10" tensordict tqdm pydantic pyyaml rich typer fastapi uvicorn maturin

# aarch64 rust (idempotent)
if [ ! -x "${CARGO_HOME}/bin/cargo" ]; then
  echo "=== rustup aarch64 into ${CARGO_HOME} ==="
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --no-modify-path --default-toolchain stable \
      --default-host aarch64-unknown-linux-gnu
fi
export PATH="${CARGO_HOME}/bin:${PATH}"
echo "cargo: $(command -v cargo) $(cargo -vV | head -1)"

echo "=== maturin develop local molrs (needs Neighbors) ==="
(
  cd "${MOLCRAFTS}/molrs/molrs-python"
  # force reinstall over any incomplete PyPI wheel
  python -m pip uninstall -y molcrafts-molrs 2>/dev/null || true
  maturin develop --release
)
python -c "import molrs; print('molrs', getattr(molrs,'__version__','?'), 'Neighbors', hasattr(molrs,'Neighbors'))"

python -m pip install -q -e "${MOLCRAFTS}/molpy"
python -m pip install -q -e "${MOLCRAFTS}/mollog" -e "${MOLCRAFTS}/molcfg" -e "${MOLCRAFTS}/molq" 2>/dev/null || true

export PYTHONPATH="${MOLNEX}/src:${MOLCRAFTS}/molhub/src:${MOLCRAFTS}/molexp/src:${MOLCRAFTS}/molpy/src:${MOLCRAFTS}/molq/src:${MOLCRAFTS}/mollog/src:${MOLCRAFTS}/molcfg/src:${PYTHONPATH:-}"

# molnex pure-Python (C++ op optional)
python -m pip install -q --no-build-isolation -e "${MOLNEX}" --no-deps 2>/dev/null || true

python - <<'PY'
import platform
import torch
print("arch", platform.machine(), "torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
from molpot.forcefield import ClassicalIMM  # noqa: F401
from molhub.dataset import PhalkethohMMDataset  # noqa: F401
import molexp  # noqa: F401
print("imports OK")
PY

WS="${MOLCRAFTS}/workspaces/mm-param-learning"
DATA="${WS}/cache/espaloma/phalkethoh-mm-small-normalized"
RUN_ID="${RUN_ID:-protocol-full-energy-mini-003}"
N_MOLS="${N_MOLS:-12}"
MAX_CONFS="${MAX_CONFS:-20}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
ENERGY_SCOPE="${ENERGY_SCOPE:-full}"

echo "=== train energy_scope=${ENERGY_SCOPE} n_mols=${N_MOLS} max_confs=${MAX_CONFS} epochs=${EPOCHS} run=${RUN_ID} ==="

python scripts/mm_param_learning/setup_class_i_protocol.py \
  --root "${WS}" \
  --run-id "${RUN_ID}" \
  --n-mols "${N_MOLS}" \
  --max-confs "${MAX_CONFS}" \
  --epochs "${EPOCHS}"

OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" \
python -u scripts/mm_param_learning/espaloma_mm_protocol_train.py \
  --data-root "${DATA}" \
  --energy-scope "${ENERGY_SCOPE}" \
  --n-mols "${N_MOLS}" \
  --max-confs "${MAX_CONFS}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --eval-every 5 \
  --device cuda \
  --workspace "${WS}" \
  --project mm-param-learning \
  --experiment class-i-full-energy-mini \
  --run-id "${RUN_ID}" \
  --dataset-coordinate dataset:espaloma/phalkethoh-mm-small@1

echo "=== DONE $(date -Iseconds) ==="
