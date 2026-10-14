#!/usr/bin/env bash
# Create the conda environment that runs everything in this repository (no root needed).
#
#   bash environment/setup_env.sh                 # -> conda env "1mm-core" (Python 3.10 + OpenFHE 1.5.1)
#   bash environment/setup_env.sh --name myenv    # another env name (then export CORE_ENV=myenv)
#   bash environment/setup_env.sh --build-tools   # also add a C++ toolchain (gcc, cmake, make, git) to the env,
#                                                 # for the optional SEAL backend of Figure 15
# Needs: conda (Miniconda/Miniforge) and internet access for pip/conda.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="${CORE_ENV:-1mm-core}"; TOOLS=0
while [ $# -gt 0 ]; do case "$1" in
  --name) NAME="$2"; shift 2;;
  --build-tools) TOOLS=1; shift;;
  -h|--help) sed -n '2,9p' "$0"; exit 0;;
  *) echo "unknown arg: $1" >&2; exit 1;; esac; done

command -v conda >/dev/null 2>&1 || { echo "ERROR: conda not found. Install Miniconda first (https://docs.conda.io)." >&2; exit 1; }
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$NAME"; then
  echo "==> conda env '$NAME' exists; installing/updating packages in it"
else
  echo "==> creating conda env '$NAME' (python 3.10; the openfhe wheels exist for 3.10 and 3.12 only)"
  # conda-forge only: a fresh Miniconda otherwise stops at Anaconda's channel Terms of Service
  conda create -y --override-channels -c conda-forge -n "$NAME" python=3.10
fi
conda activate "$NAME"
pip install --disable-pip-version-check -r "$HERE/core-requirements.txt"
if [ "$TOOLS" = 1 ]; then
  echo "==> adding the C++ toolchain for the SEAL backend"
  conda install -y --override-channels -c conda-forge cxx-compiler cmake make git
fi
python - <<'PY'
import importlib.metadata as m, openfhe, sklearn, pandas, matplotlib, numpy
print("  OK: openfhe", m.version("openfhe"), "| scikit-learn", sklearn.__version__, "| numpy", numpy.__version__)
PY
echo "Done. Activate with:  conda activate $NAME"
[ "$NAME" != "1mm-core" ] && echo "Set  export CORE_ENV=$NAME  before running the experiment scripts."
exit 0
