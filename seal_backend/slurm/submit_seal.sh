#!/usr/bin/env bash
# Submit the SEAL runs of Figure 15 as Slurm jobs (site settings as in reproduce/slurm/submit.sh).
#   seal_backend/slurm/submit_seal.sh                 # the four jobs of the paper: {bgv,bfv} x {rf models, mnist}
#   seal_backend/slurm/submit_seal.sh --check         # one 1-minute job on the irisrf6 micro model
#   seal_backend/slurm/submit_seal.sh --dry-run
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
# shellcheck disable=SC1091
[ -f reproduce/slurm/site.env ] && source reproduce/slurm/site.env
PART="${SLURM_PARTITION:-}"; MEM="${SLURM_MEM:-}"; CPUS=4; WALL="12:00:00"; DRY=0; CHECK=0; RESULTS=""; LABEL=""
while [ $# -gt 0 ]; do case "$1" in
  --partition) PART="$2"; shift 2;;  --mem) MEM="$2"; shift 2;;  --cpus) CPUS="$2"; shift 2;;  --time) WALL="$2"; shift 2;;
  --results) RESULTS="$2"; shift 2;;  --label) LABEL="$2"; shift 2;;  --check) CHECK=1; shift;;  --dry-run) DRY=1; shift;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done
[ -z "$LABEL" ] && LABEL="${PART:-slurm}"; [ -z "$RESULTS" ] && RESULTS="data/results/$LABEL"; [ "$DRY" = 1 ] || mkdir -p "$RESULTS/logs"
jobs=("se_rf bgv income,iris,soccer" "se_rfmnist bgv mnist" "se_rf bfv income,iris,soccer" "se_rfmnist bfv mnist")
[ "$CHECK" = 1 ] && { jobs=("se_micro bgv irisrf6"); WALL="00:30:00"; }
for j in "${jobs[@]}"; do
  read -r tag scheme models <<<"$j"
  args=( -J "${tag}_${scheme}" -c "$CPUS" -t "$WALL" --chdir="$REPO" -o "$RESULTS/logs/%x-%j.out"
         --export="ALL,REPO=$REPO,TAG=$tag,SCHEME=$scheme,MODELS=${models//,/ },RESULTS=$RESULTS,RESULT_LABEL=$LABEL,CORE_ENV=${CORE_ENV:-1mm-core}" )
  [ -n "$PART" ] && args+=( -p "$PART" ); [ -n "$MEM" ] && args+=( --mem="$MEM" )
  # shellcheck disable=SC2206
  [ -n "${SLURM_EXTRA_ARGS:-}" ] && args+=( ${SLURM_EXTRA_ARGS} )
  args+=( seal_backend/slurm/run_seal.sbatch )
  if [ "$DRY" = 1 ]; then echo "  DRY sbatch ${args[*]}"; else echo "  submitted ${tag}_${scheme} jobid=$(sbatch --parsable "${args[@]}")"; fi
done
