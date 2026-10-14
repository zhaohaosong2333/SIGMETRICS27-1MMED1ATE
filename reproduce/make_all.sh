#!/usr/bin/env bash
# ============================================================================
# Regenerate every data figure and table of the paper from a results directory.
#
#   bash reproduce/make_all.sh                                   # paper results -> reproduce/figures_paper/
#   bash reproduce/make_all.sh --results data/results/mylabel --outdir out/mylabel   # your own results
#
# Outputs (in --outdir), named like the paper's fig/ files:
#   multiplicativedepth.pdf   Figure 2       (tag motiv)
#   rw_ci.pdf                 Figure 11                   (tags s_/m_{rf,gbdt}_bgv_{ud,od})
#   tab_absolute_ci.md/.tex, tab_absolute_paper.tex, headline_ci.txt   Table 9 and the headline numbers
#   tab_modelinfo.md          Table 5                 (from the models only)
#   tab_compiledrep.md        Table 8               (static profiles + ciphertext sizes of the sw_* tags)
#   sweep_scaling.pdf         Figure 12               (tags sw_{ens,size,depth,full}_{od,ud})
#   sweep_model.pdf           Figure 14                (tag sw_fhe + all sw_* tags)
#   tab_sweep_*.md, sweep_fits.txt, sweep_costmodel/fheparam/membreakdown.pdf   the sensitivity-study numbers
#   threadscaling.pdf         Figure 13             (tag s_thread, --omp 1..32)
#   backends.pdf              Figure 15                  (tags s_rf_bfv_*, s_rfmnist_bfv_*, s_rf_bgv_* + seal_backend/run_seal.sh)
# Each step skips (with a message) when its data is not in the results directory.
#
# Python: the active conda env if it is $CORE_ENV, else `conda run -n $CORE_ENV`; override with $REPRO_PY.
# ============================================================================
set -uo pipefail
# Fixed timestamp for the PDF metadata so that regenerated figures are byte-identical to the committed ones.
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-1790899200}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
CORE="${CORE_ENV:-1mm-core}"
RESULTS="data/paper_results"; OUT="reproduce/figures_paper"
while [ $# -gt 0 ]; do case "$1" in
  --results) RESULTS="$2"; shift 2;;
  --outdir)  OUT="$2"; shift 2;;
  -h|--help) sed -n '3,23p' "$0" | sed 's/^#\s\{0,1\}//'; exit 0;;
  *) echo "unknown arg: $1 (see --help)" >&2; exit 2;;
esac; done
[ -d "$RESULTS" ] || { echo "ERROR: results directory '$RESULTS' does not exist" >&2; exit 1; }
mkdir -p "$OUT"

if [ -n "${REPRO_PY:-}" ]; then PY=("$REPRO_PY"); PYDESC="\$REPRO_PY"
elif [ "${CONDA_DEFAULT_ENV:-}" = "$CORE" ]; then PY=(python); PYDESC="active conda env $CORE"
elif command -v conda >/dev/null 2>&1 && conda run -n "$CORE" python -c '' >/dev/null 2>&1; then
  PY=(conda run --no-capture-output -n "$CORE" python); PYDESC="conda env $CORE"
else
  echo "ERROR: conda env '$CORE' not found; create it with  bash environment/setup_env.sh" >&2; exit 1
fi
run() { "${PY[@]}" "$@"; }
echo "== python: $PYDESC | results: $RESULTS | outputs: $OUT =="

echo "== [1/6] Figure 2 =="
run reproduce/make_figures.py --results "$RESULTS" --outdir "$OUT"
echo "== [2/6] Figure 11 + Table 9 (mean and 95% CI over the repetitions present) =="
run reproduce/make_end2end.py --results "$RESULTS" --outdir "$OUT" \
  || echo "  [skip] Figure 11 / Table 9 need the s_/m_{rf,gbdt}_bgv_{ud,od} tags"
echo "== [3/6] Table 5 (from the model files) =="
run reproduce/make_extra_tables.py --outdir "$OUT"
echo "== [4/6] Figure 12, Figure 14 and the sensitivity-study tables =="
run reproduce/make_sweep_figures.py --results "$RESULTS" --outdir "$OUT" \
  || echo "  [skip] sweep figures need the sw_* tags"
echo "== [5/6] Table 8 =="
run reproduce/make_compiledrep_table.py --results "$RESULTS" --outdir "$OUT" \
  || echo "  [skip] Table 8 needs the sw_* tags (ciphertext sizes per ring and depth)"
echo "== [6/6] Figure 15 + Figure 13 =="
run reproduce/make_new_figures.py --results "$RESULTS" --outdir "$OUT"
echo "== DONE: $OUT =="
ls -1 "$OUT" | sed 's/^/   /'
