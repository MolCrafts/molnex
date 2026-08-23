#!/bin/bash
#SBATCH --job-name=class-i-esp-scale
#SBATCH --account=naiss2026-1-53-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --output=class-i-espaloma-scale-%j.out

# Formal Class-I full-energy train at Espaloma MM-small scale:
#   173 molecules × 100 confs, ChemEncoder dim=128 n_blocks=5, Adam 1e-4,
#   batch=100, clip_grad_norm=1.0, official splits.json seed=2666,
#   energy_scope=full (bonded+EEM+LJ+Coulomb).
# Espaloma notebook budget is 10k epochs; we use 1000 + early-stop patience 50.
#
#   sbatch scripts/mm_param_learning/run_class_i_espaloma_scale_gh200.sbatch

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
export RUSTUP_HOME="${MOLNEX}/.rustup-gh200"
export CARGO_HOME="${MOLNEX}/.cargo-gh200"
export PATH="$(echo "${PATH}" | tr ':' '\n' | grep -v 'x86_64/cargo' | paste -sd: -)"
export PATH="${CARGO_HOME}/bin:${PATH}"

cd "${MOLNEX}"
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

if [ "${PREPARE_ENV:-0}" = "1" ]; then
  # Serialize the exceptional mutable setup path; ordinary jobs treat the
  # shared environment as immutable.
  exec 9>"${VENV}/.prepare.lock"
  flock 9
  python -m pip install -q --upgrade pip
  python -m pip install -q "torch>=2.10" tensordict tqdm pydantic pyyaml rich typer fastapi uvicorn maturin
fi

# Ensure local molrs with Neighbors only during explicit environment setup.
if [ "${PREPARE_ENV:-0}" = "1" ] && ! python -c "import molrs; assert hasattr(molrs,'Neighbors')" 2>/dev/null; then
  echo "=== maturin develop molrs ==="
  if [ ! -x "${CARGO_HOME}/bin/cargo" ]; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --no-modify-path --default-toolchain stable \
        --default-host aarch64-unknown-linux-gnu
  fi
  (cd "${MOLCRAFTS}/molrs/molrs-python" && python -m pip uninstall -y molcrafts-molrs 2>/dev/null || true && maturin develop --release)
fi
if [ "${PREPARE_ENV:-0}" = "1" ]; then
  python -m pip install -q -e "${MOLCRAFTS}/molpy"
  python -m pip install -q -e "${MOLCRAFTS}/mollog" -e "${MOLCRAFTS}/molcfg" -e "${MOLCRAFTS}/molq" 2>/dev/null || true
fi

export PYTHONPATH="${MOLNEX}:${MOLNEX}/src:${MOLCRAFTS}/molhub/src:${MOLCRAFTS}/molexp/src:${MOLCRAFTS}/molpy/src:${MOLCRAFTS}/molq/src:${MOLCRAFTS}/mollog/src:${MOLCRAFTS}/molcfg/src:${PYTHONPATH:-}"

python - <<'PY'
import platform, torch
print("arch", platform.machine(), "torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
from molpot.forcefield import Parameterizer
from molhub.dataset import PhalkethohMMDataset
import molexp
import scripts.mm_param_learning.espaloma_mm_protocol_train
print("imports OK")
PY

WS="${MOLCRAFTS}/workspaces/mm-param-learning"

echo "=== molexp Class-I v0 profile=full ==="

OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" \
python -m molexp.cli run scripts/mm_param_learning/train.py \
  --config scripts/mm_param_learning/molcfg.yaml \
  --profile full \
  -ws "${WS}"

echo "=== DONE $(date -Iseconds) ==="
