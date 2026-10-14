#!/usr/bin/env python3
"""Regression test for the IMM dual-feature complement encoding.

The 1MM right-edge decision (x >= t) is computed as a complement "<":
    (2^p - 1 - x) < (2^p - t)   <=>   x >= t.
Dropping the -1 turns it into x > t (strict), which is WRONG at exact ties x == t
(the right edge is then never taken). This test forces ties by setting every input
feature equal to its branch's threshold, and asserts all three protocols still agree.

It FAILS with the buggy `2^p - x` and PASSES with the correct `2^p - 1 - x`.
Requires OpenFHE (BGV). Run: python tests/test_dual_feature_tie.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from immed1ate.compiler.extractor import get_unsecure_copse_model_from_serial_model
from immed1ate.experiments import ExperimentConfig, ExperimentRunner
import immed1ate.experiments.runner as runner_mod

# Small, fast models where the number of thresholds equals the number of branches,
# so setting input = thresholds makes x == t at every comparison.
TIE_MODELS = ["irisrf6", "irisrf10", "irisrf22"]


def run_tie(model_name: str) -> bool:
    mf = str(REPO / "data" / "models" / "micro" / f"{model_name}.mdl")
    copse = get_unsecure_copse_model_from_serial_model(mf)
    tie = list(copse.thresh)
    assert len(tie) == copse.b, (
        f"{model_name}: need q==b to build a tie input ({len(tie)} vs {copse.b})")

    # Force the runtime input to equal the thresholds -> x == t at every branch.
    runner_mod.generate_random_vector = lambda n, maxval, _t=tie: list(_t)

    cfg = ExperimentConfig(
        protocols=["IMM", "VESTA", "COPSE"], bit_width=copse.p, fhe_scheme="BGV",
        depth=-1, config_setting="OD", precision=copse.p, num_runs=1,
        save_results=False, measure_preprocessing=False, measure_encryption=False,
        measure_evaluation=True, measure_decryption=True)
    cfg.model_files = [mf]
    results = ExperimentRunner(cfg).run_all_experiments()
    return bool(results and results[0].results_match)


def test_dual_feature_tie():
    for m in TIE_MODELS:
        assert run_tie(m), (
            f"IMM/VESTA/COPSE disagree at exact feature==threshold ties for {m} "
            f"(IMM dual-feature encoding must be 2^p-1-x, not 2^p-x)")


if __name__ == "__main__":
    failures = []
    for m in TIE_MODELS:
        ok = run_tie(m)
        print(f"{m:<10} tie-input  IMM==VESTA==COPSE: {ok}")
        if not ok:
            failures.append(m)
    if failures:
        print("FAIL (dual-feature bug):", failures)
        sys.exit(1)
    print("PASS: all protocols agree at exact feature==threshold ties")
