# shellcheck shell=bash
# Common environment for every 1MMED1ATE experiment process. SOURCE this; do not execute it.
# Requires $REPO (repository root). Provides:
#   - activation of the conda env $CORE_ENV (default 1mm-core) unless it is already active
#   - $THREADS: OpenMP thread count = $OMP_OVERRIDE, else the Slurm allocation, else the physical cores
#   - $NUMACTL: "numactl --interleave=all" when numactl exists (reproducible timing on multi-socket nodes)
#   - $GNU_TIME: a `time` that supports -v (the memory experiments need it), or ""
#   - ulimit -d unlimited (a small data-segment limit makes large OpenFHE allocations fail silently)
#   - record_hardware <file>: writes the hardware and software that executed the run
# Deliberately no `set -e`: `conda activate` can return nonzero benignly.

: "${REPO:?REPO must be set before sourcing env.sh}"
CORE_ENV="${CORE_ENV:-1mm-core}"

# --- conda: the env may already be active (interactive use); otherwise activate it ---
if [ "${CONDA_DEFAULT_ENV:-}" != "$CORE_ENV" ]; then
  _conda_base="${CONDA_BASE:-}"
  if [ -z "$_conda_base" ] && command -v conda >/dev/null 2>&1; then
    _conda_base="$(conda info --base 2>/dev/null)"
  fi
  if [ -z "$_conda_base" ]; then
    for _c in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge" "/opt/conda" "/opt/miniconda3"; do
      [ -f "$_c/etc/profile.d/conda.sh" ] && { _conda_base="$_c"; break; }
    done
  fi
  if [ -z "$_conda_base" ] || [ ! -f "$_conda_base/etc/profile.d/conda.sh" ]; then
    echo "ERROR(env.sh): conda not found; install Miniconda and/or set CONDA_BASE=/path/to/conda" >&2
    exit 1
  fi
  # shellcheck disable=SC1091
  source "$_conda_base/etc/profile.d/conda.sh"
  conda activate "$CORE_ENV" || {
    echo "ERROR(env.sh): cannot activate conda env '$CORE_ENV' (create it: bash environment/setup_env.sh)" >&2
    exit 1
  }
fi

# --- threads: one per physical core unless told otherwise ---
_physical_cores() {
  local n
  n="$(lscpu -p=CORE 2>/dev/null | grep -v '^#' | sort -u | wc -l)"
  [ "${n:-0}" -gt 0 ] && echo "$n" || nproc
}
export THREADS="${OMP_OVERRIDE:-${SLURM_CPUS_PER_TASK:-$(_physical_cores)}}"
export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export OMP_DYNAMIC=FALSE
export PYTHONUNBUFFERED=1

# --- scratch: honour an existing TMPDIR (set it in site.env when the node-local /tmp is small) ---
[ -n "${TMPDIR:-}" ] && mkdir -p "$TMPDIR"

# --- allow large allocations ---
ulimit -d unlimited 2>/dev/null || true

# --- NUMA interleaving for reproducible timing on multi-socket nodes (harmless on one socket) ---
if command -v numactl >/dev/null 2>&1 && numactl --hardware >/dev/null 2>&1; then
  export NUMACTL="numactl --interleave=all"
else
  export NUMACTL=""
fi

# --- a GNU `time` that supports -v ---
if /usr/bin/time -v true >/dev/null 2>&1; then
  export GNU_TIME=/usr/bin/time
elif command -v gtime >/dev/null 2>&1 && gtime -v true >/dev/null 2>&1; then
  export GNU_TIME="$(command -v gtime)"
else
  export GNU_TIME=""
fi

record_hardware() {
  local out="$1"; mkdir -p "$(dirname "$out")"
  {
    echo "# 1MMED1ATE hardware record  $(date -Is 2>/dev/null || date)"
    echo "host           $(hostname)"
    echo "slurm_job      ${SLURM_JOB_ID:-none}"
    echo "partition      ${SLURM_JOB_PARTITION:-none}"
    echo "nodelist       ${SLURM_JOB_NODELIST:-none}"
    echo "cpus_per_task  ${SLURM_CPUS_PER_TASK:-none}"
    echo "threads_used   ${THREADS:-?}"
    echo "nproc          $(nproc)"
    echo "numactl        ${NUMACTL:-<none>}"
    echo "gnu_time       ${GNU_TIME:-<none>}"
    echo "git_rev        $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "== lscpu =="; lscpu 2>/dev/null
    echo "== /proc/meminfo (MemTotal) =="; grep -E '^MemTotal' /proc/meminfo 2>/dev/null
    echo "== python / openfhe =="; python -V 2>&1
    python - <<'PY' 2>&1
import importlib.metadata as m
for p in ("openfhe", "numpy", "scikit-learn", "pandas", "matplotlib"):
    try: print(f"{p} {m.version(p)}")
    except Exception as e: print(f"{p} not installed ({e})")
PY
  } > "$out" 2>&1
  echo "[hw] recorded -> $out"
}
