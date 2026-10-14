#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run ONE experiment tag of reproduce/manifest.tsv end-to-end on the current machine:
# experiment -> collected CSV -> hardware record. This is the single implementation
# behind both execution paths:
#   plain execution : reproduce/run_local.sh      (loops over tags, runs them one after another)
#   Slurm           : reproduce/slurm/submit.sh   (one sbatch job per tag -> slurm/run_one.sbatch)
#
#   bash reproduce/run_tag.sh <tag> [--results DIR] [--label L] [--rep N] [--omp N]
#
#   --results DIR  results directory (default data/results/<label>)
#   --label L      provenance token in the output names (default: Slurm partition, else host name)
#   --rep N        repetition number (default 1); independent repetitions give the 95% CIs
#   --omp N        OpenMP threads (default: Slurm allocation, else physical cores); s_thread's output gets __omp<N>
# Outputs in DIR:  <tag>__<label>__rep<N>[__omp<T>].csv (speed, sweep) | .mem.csv + .wide.csv (memory)
#                  sweep_raw/<...>.jsonl (sweep), motivation_matmul_depth.txt (motivation),
#                  hwlog/<...>.txt, logs/<...>.log, _run/<...>/ (scratch of the run)
# ---------------------------------------------------------------------------
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

TAG="${1:-}"; [ -z "$TAG" ] && { sed -n '2,20p' "$0"; exit 2; }; shift
REP="${REP:-1}"; RESULTS="${RESULTS:-}"; LABEL="${RESULT_LABEL:-}"
while [ $# -gt 0 ]; do case "$1" in
  --results) RESULTS="$2"; shift 2;;
  --label)   LABEL="$2"; shift 2;;
  --rep)     REP="$2"; shift 2;;
  --omp)     export OMP_OVERRIDE="$2"; shift 2;;
  *) echo "unknown arg: $1" >&2; exit 2;; esac; done

# shellcheck disable=SC1091
source reproduce/env.sh

ROW="$(awk -v t="$TAG" '$1!~/^#/ && $1==t {print; exit}' reproduce/manifest.tsv)"
[ -z "$ROW" ] && { echo "ERROR: tag '$TAG' not found in reproduce/manifest.tsv"; exit 2; }
IFS=$'\t' read -r _ FAMILY MODEL_FOLDER MODELS SCHEME CONFIG _CPUS _WALL <<<"$ROW"

[ -z "$LABEL" ] && LABEL="${SLURM_JOB_PARTITION:-$(hostname -s)}"
[ -z "$RESULTS" ] && RESULTS="data/results/$LABEL"
SUFFIX="${LABEL}__rep${REP}"
# only the thread-scaling tag keeps one file per thread count; for every other tag the thread count
# is a property of the results directory (recorded in hwlog/), so the file name stays the same
[ -n "${OMP_OVERRIDE:-}" ] && [ "$TAG" = "s_thread" ] && SUFFIX="${SUFFIX}__omp${OMP_OVERRIDE}"
NAME="${TAG}__${SUFFIX}"
RUNDIR="$RESULTS/_run/$NAME"
rm -rf "$RUNDIR"; mkdir -p "$RUNDIR" "$RESULTS/hwlog" "$RESULTS/logs" "$RESULTS/sweep_raw"
record_hardware "$RESULTS/hwlog/$NAME.txt"

MODEL_ARGS=()
[ "$MODELS" != "-" ] && MODEL_ARGS=(--models ${MODELS//,/ })

echo "=================================================================="
echo " RUN tag=$TAG family=$FAMILY models=$MODEL_FOLDER:$MODELS scheme=$SCHEME config=$CONFIG"
echo " host=$(hostname) threads=$THREADS rep=$REP results=$RESULTS  $(date -Is 2>/dev/null || date)"
echo "=================================================================="

rc=0
case "$FAMILY" in
  speed)
    # one process per model (a fresh OpenFHE context each time); the collector merges the per-model logs
    if [ "$MODELS" != "-" ]; then list=${MODELS//,/ }; else list=$(ls "$MODEL_FOLDER"/*.mdl | xargs -n1 basename | sed 's/\.mdl$//'); fi
    for m in $list; do
      $NUMACTL python scripts/run_speed_experiment.py --model-folder "$MODEL_FOLDER" --models "$m" \
          --fhe-scheme "$SCHEME" --config-setting "$CONFIG" --output-folder "$RUNDIR" || rc=$?
      sleep 1   # the log file name carries a one-second timestamp
    done
    python reproduce/collect_speed_csv.py "$RUNDIR" --out "$RESULTS/$NAME.csv" || rc=$?
    ;;
  memory)
    [ -z "$GNU_TIME" ] && echo "WARNING: no GNU time with -v found; peak memory will be missing"
    $NUMACTL python scripts/run_complete_experiment.py --model-folder "$MODEL_FOLDER" "${MODEL_ARGS[@]}" \
        --fhe-scheme "$SCHEME" --config-setting "$CONFIG" \
        --output-folder "$RUNDIR" --output-file "$RUNDIR/full.json" || rc=$?
    python reproduce/collect_memory_csv.py "$RUNDIR" --out "$RESULTS/$NAME.mem.csv" || rc=$?
    python reproduce/collect_memory_wide_csv.py "$RUNDIR" --setting "$CONFIG" --out "$RESULTS/$NAME.wide.csv" || rc=$?
    ;;
  sweep)
    # one GNU-time'd process per (model, protocol[, depth]); appends to the raw JSONL and resumes
    # from it, so re-running a tag after a time-out only runs what is missing
    $NUMACTL python scripts/run_sweep_experiment.py --model-folder "$MODEL_FOLDER" "${MODEL_ARGS[@]}" \
        --config-setting "$CONFIG" --output-folder "$RUNDIR" \
        --jsonl "$RESULTS/sweep_raw/$NAME.jsonl" --resume || rc=$?
    python reproduce/collect_sweep_csv.py "$RESULTS/sweep_raw/$NAME.jsonl" --out "$RESULTS/$NAME.csv" || rc=$?
    ;;
  motivation)
    $NUMACTL python scripts/run_motivation_experiment.py --out "$RESULTS/motivation_matmul_depth.txt" || rc=$?
    ;;
  *) echo "ERROR: unknown family '$FAMILY'"; exit 3;;
esac

echo "=================================================================="
echo " DONE tag=$TAG rc=$rc  $(date -Is 2>/dev/null || date)"
echo "=================================================================="
exit "$rc"
