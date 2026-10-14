#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run the SEAL backend (1MM, VESTA, COPSE at OD and UD) over a set of exported models and collect the
# stage-timing CSVs that Figure 15 reads. Single-threaded (SEAL's core has no thread knob; HEXL uses
# SIMD on one core), so these runs are long: the four RF models take about 2 h (bgv) and 4 h (bfv).
#
#   bash seal_backend/run_seal.sh --tag se_micro   --scheme bgv --models irisrf6               # 1-minute check
#   bash seal_backend/run_seal.sh --tag se_rf      --scheme bgv --models "income iris soccer"  # Figure 15
#   bash seal_backend/run_seal.sh --tag se_rfmnist --scheme bgv --models mnist
#   (and the same two with --scheme bfv)
#
#   --models M     model stems; looked up in data/models/rf (or data/models/micro for irisrf*)
#   --configs C    "OD UD" (default) or one of them
#   --results DIR  results directory (default data/results/<label>);  --label L  (default: host name)
#   --timeout S    per-bench time limit in seconds (default 21600)
# Outputs: <results>/<tag>_<scheme>_<CFG>__<label>__rep1.csv, <results>/hwlog/<tag>_<scheme>__<label>.txt
# The exported tensors (data/exported/<model>.json) are produced on demand with scripts/export_tensors.py
# (needs the conda env), so the C++ benches evaluate exactly the models the OpenFHE runs used.
# ---------------------------------------------------------------------------
set -uo pipefail
SB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$SB")"; cd "$REPO"
TAG=""; SCHEME="bgv"; MODELS=""; CONFIGS="OD UD"; RESULTS=""; LABEL=""; TO="${PERBENCH_TIMEOUT:-21600}"
while [ $# -gt 0 ]; do case "$1" in
  --tag) TAG="$2"; shift 2;;  --scheme) SCHEME="$2"; shift 2;;  --models) MODELS="$2"; shift 2;;
  --configs) CONFIGS="$2"; shift 2;;  --results) RESULTS="$2"; shift 2;;  --label) LABEL="$2"; shift 2;;
  --timeout) TO="$2"; shift 2;;  -h|--help) sed -n '2,18p' "$0"; exit 0;;
  *) echo "unknown arg: $1" >&2; exit 2;; esac; done
[ -n "$TAG" ] && [ -n "$MODELS" ] || { echo "need --tag and --models (see --help)" >&2; exit 2; }
BE="$SB/build"
# (a build that finished seconds ago may not be visible yet on a network file system: wait briefly)
for _i in $(seq 1 12); do [ -x "$BE/imm_bench" ] && break; sleep 5; done
[ -x "$BE/imm_bench" ] || { echo "ERROR: benches not built; run bash seal_backend/build.sh" >&2; exit 1; }
[ -z "$LABEL" ] && LABEL="${RESULT_LABEL:-${SLURM_JOB_PARTITION:-$(hostname -s)}}"
[ -z "$RESULTS" ] && RESULTS="data/results/$LABEL"
mkdir -p "$RESULTS/hwlog" "$RESULTS/logs"
ulimit -d unlimited 2>/dev/null || true

# the python side (export + collect) runs in the conda env
# shellcheck disable=SC1091
source reproduce/env.sh
export OMP_NUM_THREADS=1

# export tensors for any model that lacks them
need=()
for m in $MODELS; do [ -f "data/exported/$m.json" ] || need+=("$m"); done
if [ "${#need[@]}" -gt 0 ]; then
  folder=data/models/rf; case "${need[0]}" in irisrf*) folder=data/models/micro;; esac
  echo "== exporting tensors for: ${need[*]} (from $folder)"
  python scripts/export_tensors.py --model-folder "$folder" --out-dir data/exported --only "$(IFS=,; echo "${need[*]}")" || exit 1
fi

HW="$RESULTS/hwlog/${TAG}_${SCHEME}__${LABEL}.txt"
{ echo "# seal-run tag=$TAG scheme=$SCHEME models=$MODELS  $(date -Is)  host $(hostname)  slurm_job ${SLURM_JOB_ID:-none}";
  lscpu 2>/dev/null | grep -E 'Model name|Flags' | sed 's/Flags.*avx512.*/Flags: (avx512 present)/'; grep MemTotal /proc/meminfo;
  echo "SEAL: $(grep -h 'SEAL_VERSION' "$SB/_deps/install/include/SEAL-4.1/seal/util/config.h" 2>/dev/null | head -1)";
  echo "HEXL: $(grep -c 'SEAL_USE_INTEL_HEXL' "$SB/_deps/install/include/SEAL-4.1/seal/util/config.h" 2>/dev/null) (1 = on)"; } > "$HW"
for config in $CONFIGS; do
  RUN="$RESULTS/_run/${TAG}_${SCHEME}_${config}__${LABEL}"; rm -rf "$RUN"; mkdir -p "$RUN"
  for name in $MODELS; do
    J="data/exported/$name.json"; [ -f "$J" ] || { echo "MISSING $J, skip"; continue; }
    for pb in imm vesta copse; do
      echo ">>> $TAG $name $pb $config $(date +%H:%M:%S)"; t0=$SECONDS
      timeout "$TO" "$BE/${pb}_bench" --exported "$J" --config "$config" --out "$RUN" --threads 1 --scheme "$SCHEME" \
        2>&1 | tee -a "$RESULTS/logs/${TAG}_${SCHEME}_${config}__${LABEL}.log" | grep -E "golden=|rc=" | sed 's/^/    /'
      echo "    elapsed=$((SECONDS - t0))s"
    done
  done
  python reproduce/collect_speed_csv.py "$RUN" --out "$RESULTS/${TAG}_${SCHEME}_${config}__${LABEL}__rep1.csv"
done
echo "== done $(date -Is)"
