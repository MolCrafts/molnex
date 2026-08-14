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

export PYTHONPATH="${MOLNEX}/src:${MOLCRAFTS}/molhub/src:${MOLCRAFTS}/molexp/src:${MOLCRAFTS}/molpy/src:${MOLCRAFTS}/molq/src:${MOLCRAFTS}/mollog/src:${MOLCRAFTS}/molcfg/src:${PYTHONPATH:-}"

python - <<'PY'
import platform, torch
print("arch", platform.machine(), "torch", torch.__version__, "cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
from molpot.forcefield import ClassicalIMM
from molhub.dataset import PhalkethohMMDataset
import molexp
import scripts.mm_param_learning.espaloma_mm_protocol_train
print("imports OK")
PY

WS="${MOLCRAFTS}/workspaces/mm-param-learning"
DATA="${WS}/cache/espaloma/phalkethoh-mm-small-normalized"
EXP="${EXP:-class-i-full-energy-espaloma-scale}"
RUN_ID="${RUN_ID:-protocol-full-energy-espaloma-scale-001}"
N_MOLS="${N_MOLS:-0}"           # 0 = all 173
MAX_CONFS="${MAX_CONFS:-100}"
EPOCHS="${EPOCHS:-1000}"
BATCH_SIZE="${BATCH_SIZE:-100}"
LR="${LR:-1e-4}"
PATIENCE="${PATIENCE:-50}"
ENERGY_SCOPE="${ENERGY_SCOPE:-full}"
HIDDEN="${HIDDEN:-128}"
N_BLOCKS="${N_BLOCKS:-5}"

echo "=== FORMAL Espaloma-scale: hidden=${HIDDEN} blocks=${N_BLOCKS} n_mols=${N_MOLS} confs=${MAX_CONFS} epochs=${EPOCHS} batch=${BATCH_SIZE} patience=${PATIENCE} ==="

# Register experiment + run under molexp (Python API)
python - <<PY
from pathlib import Path
from molexp.workspace import Workspace, SourceRef, write_knowledge_item

ws = Workspace(Path("${WS}"), name="mm-param-learning")
ws.materialize()
project = ws.add_project("mm-param-learning")
project.materialize()
params = {
    "dataset_coordinate": "dataset:espaloma/phalkethoh-mm-small@1",
    "data_root": "${DATA}",
    "energy_scope": "${ENERGY_SCOPE}",
    "program": "mm-param-learning-baseline",
    "protocol": "espaloma-mm-small-notebook",
    "n_mols": ${N_MOLS},
    "max_confs": ${MAX_CONFS},
    "epochs": ${EPOCHS},
    "lr": ${LR},
    "batch_size": ${BATCH_SIZE},
    "seed": 2666,
    "hidden": ${HIDDEN},
    "n_blocks": ${N_BLOCKS},
    "n_torsion_terms": 6,
    "early_stop_patience": ${PATIENCE},
    "use_official_splits": True,
    "train_script": "scripts/mm_param_learning/espaloma_mm_protocol_train.py",
}
sweep_space = {
    "model_sizes": {
        "small": {"hidden": 64, "n_blocks": 5, "packed_batch_size": 4096},
        "medium": {"hidden": 512, "n_blocks": 5, "packed_batch_size": 2048},
        "large": {"hidden": 4096, "n_blocks": 5, "packed_batch_size": 256},
    },
    "dataset_coordinate": "dataset:espaloma/phalkethoh-mm-small@1",
    "calibration_experiment": "class-i-packed-batch-calibration-gh200",
}
exp = project.add_experiment(
    "${EXP}",
    params=sweep_space,
    description="Power-of-two Class-I model-size sweep with calibrated packed batches",
    tags=["mm-param-learning", "full-energy", "class-i", "espaloma-scale", "packed-batch", "formal"],
)
exp.metadata = exp.metadata.model_copy(
    update={"description": exp.description, "tags": list(exp.tags), "parameter_space": sweep_space}
)
exp.save()
exp.materialize()
if exp.has_run("${RUN_ID}"):
    run = exp.get_run("${RUN_ID}")
    run._update_metadata(parameters=params)
    run.materialize()
else:
    run = exp.add_run(params=params, id="${RUN_ID}")
    run.materialize()
if "${MODEL_SIZE:-custom}" == "small":
    write_knowledge_item(
        project,
        name="class-i-power2-packed-size-sweep-protocol",
        kind="ProtocolNote",
        sources=[
            SourceRef(kind="experiment", ref="${EXP}"),
            SourceRef(kind="experiment", ref="class-i-packed-batch-calibration-gh200"),
            SourceRef(kind="file", ref="dataset:espaloma/phalkethoh-mm-small@1"),
            SourceRef(kind="reference", ref="DOI:10.1039/D2SC02739A"),
        ],
        created_by="mm-param-learning/run_class_i_espaloma_scale_gh200",
        body=(
            "# Power-of-two packed Class-I size sweep\\n\\n"
            "- Dataset: 173 mol × 100 conf, official seed 2666 splits\\n"
            "- Models: h64/b4096, h512/b2048, h4096/b256; n_blocks=5\\n"
            "- Batch sizes selected by GH200 packed forward/backward/Adam calibration\\n"
            "- Opt: Adam 1e-4, epochs=${EPOCHS}, early_stop=${PATIENCE}\\n"
        ),
        cite=[(exp, "derived_from")],
        title="Power-of-two packed Class-I size sweep protocol",
    )
print("run_dir", run.path())
PY

OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" \
python -u scripts/mm_param_learning/espaloma_mm_protocol_train.py \
  --data-root "${DATA}" \
  --energy-scope "${ENERGY_SCOPE}" \
  --n-mols "${N_MOLS}" \
  --max-confs "${MAX_CONFS}" \
  --selection all \
  --use-official-splits \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --hidden "${HIDDEN}" \
  --n-blocks "${N_BLOCKS}" \
  --lr "${LR}" \
  --eval-every 5 \
  --early-stop-patience "${PATIENCE}" \
  --device cuda \
  --workspace "${WS}" \
  --project mm-param-learning \
  --experiment "${EXP}" \
  --run-id "${RUN_ID}" \
  --dataset-coordinate dataset:espaloma/phalkethoh-mm-small@1

echo "=== DONE $(date -Iseconds) ==="
