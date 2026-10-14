#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Slurm path: submit experiment tags of reproduce/manifest.tsv as one job each. Run this on a
# login node; the work runs on the allocated compute node through slurm/run_one.sbatch ->
# reproduce/run_tag.sh (the same runner as the plain execution path).
#
# Site settings come from reproduce/slurm/site.env (copy site.env.example) or the environment:
#   SLURM_PARTITION   partition to use (omit -> the cluster's default partition)
#   SLURM_MEM         --mem per job, e.g. 490G (omit -> the partition's default; the largest run,
#                     COPSE on the MNIST RF, peaks at 96 GB)
#   SLURM_EXTRA_ARGS  anything else sbatch needs, e.g. "--account=myproj --mail-user=me@x.org --mail-type=END,FAIL"
#   CORE_ENV, CONDA_BASE, TMPDIR  as in environment/setup_env.sh and reproduce/env.sh
#
#   reproduce/slurm/submit.sh --tags smoke,sw_smoke                 # ~5 min check of both harnesses
#   reproduce/slurm/submit.sh --family speed                        # every speed tag
#   reproduce/slurm/submit.sh --tags s_rf_bgv_od --reps 3           # three independent allocations
#   reproduce/slurm/submit.sh --tags s_thread --omp 8               # one thread-scaling point
#   reproduce/slurm/submit.sh --all                                 # everything the paper measured
#   reproduce/slurm/submit.sh --all --dry-run                       # print the sbatch lines only
#
#   --partition P / --mem M / --cpus C   override the site settings / the manifest's cpus
#   --no-exclusive                       do not request --exclusive (the paper's runs were exclusive)
#   --results DIR / --label L            results directory (default data/results/<partition>)
# ---------------------------------------------------------------------------
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
MANIFEST="reproduce/manifest.tsv"
# shellcheck disable=SC1091
[ -f reproduce/slurm/site.env ] && source reproduce/slurm/site.env

FAMILY=""; TAGS=""; ALL=0; REPS=1; OMP=""; DRY=0; RESULTS=""; LABEL=""; EXCL=1
PART="${SLURM_PARTITION:-}"; MEM="${SLURM_MEM:-}"; CPUS=""
while [ $# -gt 0 ]; do case "$1" in
  --family)       FAMILY="$2"; shift 2;;
  --tags)         TAGS="$2"; shift 2;;
  --all)          ALL=1; shift;;
  --reps)         REPS="$2"; shift 2;;
  --omp)          OMP="$2"; shift 2;;
  --partition)    PART="$2"; shift 2;;
  --mem)          MEM="$2"; shift 2;;
  --cpus)         CPUS="$2"; shift 2;;
  --no-exclusive) EXCL=0; shift;;
  --results)      RESULTS="$2"; shift 2;;
  --label)        LABEL="$2"; shift 2;;
  --dry-run)      DRY=1; shift;;
  -h|--help)      sed -n '2,26p' "$0"; exit 0;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done

declare -a TAGLIST=()
if [ -n "$TAGS" ]; then IFS=',' read -ra TAGLIST <<<"$TAGS"
elif [ -n "$FAMILY" ]; then mapfile -t TAGLIST < <(awk -F'\t' -v f="$FAMILY" '$1!~/^#/ && NF>=8 && $2==f {print $1}' "$MANIFEST")
elif [ "$ALL" = 1 ]; then mapfile -t TAGLIST < <(awk -F'\t' '$1!~/^#/ && NF>=8 && $1!="smoke" && $1!="sw_smoke" && $1!="s_thread" {print $1}' "$MANIFEST")
else echo "nothing to do: pass --family, --tags, or --all" >&2; exit 1; fi
[ "${#TAGLIST[@]}" -eq 0 ] && { echo "no matching tags" >&2; exit 1; }

[ -z "$LABEL" ] && LABEL="${PART:-slurm}"
[ -z "$RESULTS" ] && RESULTS="data/results/$LABEL"
[ "$DRY" = 1 ] || mkdir -p "$RESULTS/logs"
echo "tags=${TAGLIST[*]}  reps=$REPS  omp=${OMP:-auto}  partition=${PART:-<default>}  mem=${MEM:-<default>}  results=$RESULTS  $( [ "$DRY" = 1 ] && echo '(DRY-RUN)')"

submit() {  # tag rep omp
  local tag="$1" r="$2" omp="$3" row cpus wall jname
  row="$(awk -F'\t' -v t="$tag" '$1!~/^#/ && $1==t {print; exit}' "$MANIFEST")"
  [ -z "$row" ] && { echo "  skip: unknown tag '$tag'"; return; }
  IFS=$'\t' read -r _ _ _ _ _ _ cpus wall <<<"$row"
  cpus="${CPUS:-$cpus}"
  jname="${tag}__${LABEL}__r${r}"; [ -n "$omp" ] && [ "$tag" = "s_thread" ] && jname="${jname}__omp$omp"
  local exports="ALL,REPO=$REPO,TAG=$tag,REP=$r,RESULTS=$RESULTS,RESULT_LABEL=$LABEL,CORE_ENV=${CORE_ENV:-1mm-core}"
  [ -n "$omp" ] && exports="$exports,OMP_OVERRIDE=$omp"
  local args=( -J "$jname" -c "$cpus" -t "$wall" --chdir="$REPO" -o "$RESULTS/logs/%x-%j.out" --export="$exports" )
  [ -n "$PART" ] && args+=( -p "$PART" )
  [ -n "$MEM" ]  && args+=( --mem="$MEM" )
  [ "$EXCL" = 1 ] && args+=( --exclusive )
  # shellcheck disable=SC2206
  [ -n "${SLURM_EXTRA_ARGS:-}" ] && args+=( ${SLURM_EXTRA_ARGS} )
  args+=( reproduce/slurm/run_one.sbatch )
  if [ "$DRY" = 1 ]; then echo "  DRY sbatch ${args[*]}"
  else jid="$(sbatch --parsable "${args[@]}")"; echo "  submitted $jname  jobid=$jid  (${cpus} cores, t=$wall)"; fi
}
for tag in "${TAGLIST[@]}"; do
  for r in $(seq 1 "$REPS"); do submit "$tag" "$r" "$OMP"; done
done
# --all also submits the thread-scaling sweep (one job per thread count)
if [ "$ALL" = 1 ] && [ -z "$OMP" ]; then
  for t in 1 2 4 8 16 32; do submit s_thread 1 "$t"; done
fi
echo "== when the jobs are done:  bash reproduce/make_all.sh --results $RESULTS --outdir out/$LABEL"
