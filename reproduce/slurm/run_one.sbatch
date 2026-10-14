#!/bin/bash
# One Slurm job = one experiment tag. Submitted by reproduce/slurm/submit.sh, which passes the
# partition, cores, memory, time limit and the TAG/REP/RESULTS/RESULT_LABEL/OMP_OVERRIDE exports.
# All it does is call the shared runner, so the Slurm and plain paths cannot drift apart.
set -uo pipefail
cd "${REPO:?REPO must be exported (use submit.sh)}"
TAG="${TAG:?TAG must be exported (use submit.sh)}"
exec bash reproduce/run_tag.sh "$TAG" --rep "${REP:-1}" ${RESULTS:+--results "$RESULTS"} ${RESULT_LABEL:+--label "$RESULT_LABEL"} ${OMP_OVERRIDE:+--omp "$OMP_OVERRIDE"}
