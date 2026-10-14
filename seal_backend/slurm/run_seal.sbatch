#!/bin/bash
# One Slurm job = one SEAL run (tag x scheme). Submitted by seal_backend/slurm/submit_seal.sh.
set -uo pipefail
cd "${REPO:?REPO must be exported (use submit_seal.sh)}"
exec bash seal_backend/run_seal.sh --tag "${TAG:?}" --scheme "${SCHEME:?}" --models "${MODELS:?}" \
     ${RESULTS:+--results "$RESULTS"} ${RESULT_LABEL:+--label "$RESULT_LABEL"}
