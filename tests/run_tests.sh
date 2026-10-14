#!/usr/bin/env bash
# Run the regression tests (needs the 1mm-core env; the OpenFHE tests use small insecure rings).
#   bash tests/run_tests.sh            # all four (a few minutes)
#   bash tests/run_tests.sh --quick    # the two fast ones
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
tests=(test_compiler_vectorized test_matmul_replication test_imm_tiling test_dual_feature_tie)
[ "${1:-}" = "--quick" ] && tests=(test_compiler_vectorized test_matmul_replication)
rc=0
for t in "${tests[@]}"; do
  echo "=== $t"
  if python "tests/$t.py" > "tests/$t.log" 2>&1; then echo "    PASS"; else echo "    FAIL (see tests/$t.log)"; rc=1; fi
done
exit $rc
