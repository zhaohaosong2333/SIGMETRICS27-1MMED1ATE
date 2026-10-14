#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Plain execution path (no scheduler): run experiment tags of reproduce/manifest.tsv one after
# another on this machine. Same selection flags as the Slurm path (reproduce/slurm/submit.sh).
#
#   bash reproduce/run_local.sh --tags smoke,sw_smoke              # ~5 min check of both harnesses
#   bash reproduce/run_local.sh --family speed                      # every speed tag
#   bash reproduce/run_local.sh --tags s_rf_bgv_od --reps 3         # three repetitions (95% CIs)
#   bash reproduce/run_local.sh --tags s_thread --omp 8             # one thread-scaling point
#   bash reproduce/run_local.sh --all                               # everything the paper measured
#   bash reproduce/run_local.sh --all --dry-run                     # list what would run
#
#   --results DIR  results directory (default data/results/<label>; label = host name)
#   --label L      provenance token in the output file names (default: host name)
#   --omp N        OpenMP threads (default: physical cores)
# Run it detached when it will take hours:  nohup bash reproduce/run_local.sh --all > run_all.log 2>&1 &
# Then draw the figures:  bash reproduce/make_all.sh --results data/results/<label> --outdir out/<label>
# ---------------------------------------------------------------------------
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
MANIFEST="reproduce/manifest.tsv"

FAMILY=""; TAGS=""; ALL=0; REPS=1; OMP=""; DRY=0; RESULTS=""; LABEL=""
while [ $# -gt 0 ]; do case "$1" in
  --family)  FAMILY="$2"; shift 2;;
  --tags)    TAGS="$2"; shift 2;;
  --all)     ALL=1; shift;;
  --reps)    REPS="$2"; shift 2;;
  --omp)     OMP="$2"; shift 2;;
  --results) RESULTS="$2"; shift 2;;
  --label)   LABEL="$2"; shift 2;;
  --dry-run) DRY=1; shift;;
  -h|--help) sed -n '2,20p' "$0"; exit 0;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done

declare -a TAGLIST=()
if [ -n "$TAGS" ]; then IFS=',' read -ra TAGLIST <<<"$TAGS"
elif [ -n "$FAMILY" ]; then mapfile -t TAGLIST < <(awk -F'\t' -v f="$FAMILY" '$1!~/^#/ && NF>=8 && $2==f {print $1}' "$MANIFEST")
elif [ "$ALL" = 1 ]; then mapfile -t TAGLIST < <(awk -F'\t' '$1!~/^#/ && NF>=8 && $1!="smoke" && $1!="sw_smoke" && $1!="s_thread" {print $1}' "$MANIFEST")
else echo "nothing to do: pass --family, --tags, or --all" >&2; exit 1; fi
[ "${#TAGLIST[@]}" -eq 0 ] && { echo "no matching tags" >&2; exit 1; }

[ -z "$LABEL" ] && LABEL="$(hostname -s)"
[ -z "$RESULTS" ] && RESULTS="data/results/$LABEL"
[ "$DRY" = 1 ] || mkdir -p "$RESULTS/logs"
echo "tags=${TAGLIST[*]}  reps=$REPS  omp=${OMP:-auto}  results=$RESULTS  $( [ "$DRY" = 1 ] && echo '(DRY-RUN)')"

# --all also runs the thread-scaling sweep (one run per thread count, below the core count)
extra=()
if [ "$ALL" = 1 ] && [ -z "$OMP" ]; then
  ncores="$(lscpu -p=CORE 2>/dev/null | grep -v '^#' | sort -u | wc -l)"; [ "${ncores:-0}" -gt 0 ] || ncores="$(nproc)"
  for t in 1 2 4 8 16 32; do [ "$t" -le "$ncores" ] && extra+=("s_thread:$t"); done
fi

fail=0
run_one() {  # tag rep omp
  local tag="$1" rep="$2" omp="$3" name log
  name="${tag}__${LABEL}__rep${rep}"; [ -n "$omp" ] && [ "$tag" = "s_thread" ] && name="${name}__omp$omp"; log="$RESULTS/logs/$name.log"
  echo "== [$(date '+%F %T')] $name"
  if [ "$DRY" = 1 ]; then return; fi
  bash reproduce/run_tag.sh "$tag" --results "$RESULTS" --label "$LABEL" --rep "$rep" ${omp:+--omp "$omp"} > "$log" 2>&1 \
    && echo "   done  (log: $log)" || { echo "   FAILED rc=$? (log: $log)"; fail=1; }
}
for tag in "${TAGLIST[@]}"; do
  awk -F'\t' -v t="$tag" '$1!~/^#/ && $1==t {found=1} END {exit !found}' "$MANIFEST" || { echo "  skip: unknown tag '$tag'"; continue; }
  for r in $(seq 1 "$REPS"); do run_one "$tag" "$r" "$OMP"; done
done
for e in "${extra[@]}"; do run_one "${e%%:*}" 1 "${e##*:}"; done
echo "== all tags finished (results in $RESULTS); draw with: bash reproduce/make_all.sh --results $RESULTS --outdir out/$LABEL"
exit "$fail"
