#!/bin/bash
set -euo pipefail
echo "=== node $(hostname) arch $(uname -m) $(date -Iseconds) ==="
echo "=== SLURM_JOB_ID=${SLURM_JOB_ID:-} CPUS=${SLURM_CPUS_PER_TASK:-}${SLURM_NTASKS:-} ==="
MOLCRAFTS=/nobackup/proj/disk/teoroo/personal/jicli594/work/molcrafts
MOLNEX="${MOLCRAFTS}/molnex"
PY=/nobackup/proj/disk/teoroo/personal/jicli594/work/.x86_64/bin/python
export PYTHONPATH="${MOLNEX}/src:${MOLCRAFTS}/molhub/src:${MOLCRAFTS}/molpy/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
cd "${MOLNEX}"
echo "=== python ${PY} ==="
"${PY}" -c "import sys,platform; print(sys.executable, platform.machine())"
NPROC="${SLURM_CPUS_PER_TASK:-${SLURM_NTASKS:-16}}"
WORKERS=$(( NPROC > 2 ? NPROC - 2 : NPROC ))
echo "=== pytest -n ${WORKERS} ==="
exec "${PY}" -m pytest tests/ -q -n "${WORKERS}" --dist worksteal --tb=line
