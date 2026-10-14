#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Build the SEAL backend of Figure 15 without root: Microsoft SEAL 4.1.2 with Intel HEXL 1.2.5
# (fetched by SEAL's own CMake), installed under seal_backend/_deps/, then the three benches.
#
#   bash seal_backend/build.sh                 # fetch + build + self-test (10-20 min on 8 cores)
#   bash seal_backend/build.sh --fetch-only    # only clone SEAL and configure (needs internet; e.g. on a login node)
#   bash seal_backend/build.sh --no-fetch      # compile from an earlier --fetch-only (no internet needed)
#   bash seal_backend/build.sh --no-hexl       # build SEAL without HEXL (slower; the paper's numbers use HEXL)
#   JOBS=16 bash seal_backend/build.sh         # parallel build jobs (default: nproc)
#
# Toolchain: a C++17 compiler (g++ >= 9 or clang >= 10), cmake >= 3.16, git, make. If the machine has
# none, `bash environment/setup_env.sh --build-tools` adds them to the conda env (then run this script
# with that env active). The AVX-512 paths of HEXL need an AVX-512 CPU at run time; it falls back otherwise.
# ---------------------------------------------------------------------------
set -euo pipefail
SB="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$SB")"
DEPS="$SB/_deps"; SRC="$DEPS/SEAL"; PREFIX="$DEPS/install"
SEAL_TAG="${SEAL_TAG:-v4.1.2}"; HEXL=ON; FETCH=1; BUILD=1; JOBS="${JOBS:-$(nproc)}"
while [ $# -gt 0 ]; do case "$1" in
  --fetch-only) BUILD=0; shift;;
  --no-fetch)   FETCH=0; shift;;
  --no-hexl)    HEXL=OFF; shift;;
  -h|--help)    sed -n '2,17p' "$0"; exit 0;;
  *) echo "unknown arg: $1" >&2; exit 2;; esac; done
# CMake >= 4 refuses the old cmake_minimum_required of SEAL's bundled cpu_features; this (env + cache
# variable, so that nested configures see it too) keeps the policy floor at 3.5. Ignored by older CMake.
export CMAKE_POLICY_VERSION_MINIMUM=3.5
for t in cmake git make "${CXX:-c++}"; do command -v "$t" >/dev/null 2>&1 || { echo "ERROR: '$t' not found (see --help)" >&2; exit 1; }; done
echo "== compiler: $("${CXX:-c++}" --version | head -1) | cmake: $(cmake --version | head -1) | HEXL=$HEXL | jobs=$JOBS"

if [ "$FETCH" = 1 ]; then
  mkdir -p "$DEPS"
  if [ ! -d "$SRC/.git" ]; then
    echo "== cloning Microsoft SEAL $SEAL_TAG -> $SRC"
    git -c advice.detachedHead=false clone --quiet --depth 1 --branch "$SEAL_TAG" https://github.com/microsoft/SEAL.git "$SRC"
  fi
  echo "== configuring SEAL (fetches Intel HEXL when HEXL=ON)"
  cmake -S "$SRC" -B "$SRC/build" -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
      -DSEAL_USE_INTEL_HEXL="$HEXL" -DSEAL_BUILD_DEPS=ON -DSEAL_USE_MSGSL=OFF -DSEAL_USE_ZLIB=OFF -DSEAL_USE_ZSTD=OFF \
      -DSEAL_BUILD_EXAMPLES=OFF -DSEAL_BUILD_TESTS=OFF -DSEAL_BUILD_BENCH=OFF -DBUILD_SHARED_LIBS=OFF \
      -DSEAL_THROW_ON_TRANSPARENT_CIPHERTEXT=ON > "$DEPS/configure.log" 2>&1 || { tail -30 "$DEPS/configure.log"; exit 1; }
  echo "   configured (log: $DEPS/configure.log)"
fi
[ "$BUILD" = 1 ] || { echo "== fetch-only done; compile later with: bash seal_backend/build.sh --no-fetch"; exit 0; }
[ -d "$SRC/build" ] || { echo "ERROR: $SRC/build missing; run without --no-fetch first" >&2; exit 1; }

echo "== building SEAL ($JOBS jobs)"
cmake --build "$SRC/build" -j"$JOBS" > "$DEPS/build.log" 2>&1 || { tail -30 "$DEPS/build.log"; exit 1; }
cmake --install "$SRC/build" > "$DEPS/install.log" 2>&1
SEALCFG="$(find "$PREFIX" -name SEALConfig.cmake | head -1)"
[ -n "$SEALCFG" ] || { echo "ERROR: SEALConfig.cmake not found under $PREFIX" >&2; exit 1; }

echo "== building the benches (imm_bench, vesta_bench, copse_bench, core_test)"
rm -rf "$SB/build"
cmake -S "$SB" -B "$SB/build" -DSEAL_DIR="$(dirname "$SEALCFG")" -DCMAKE_BUILD_TYPE=Release > "$DEPS/bench_configure.log" 2>&1 \
  || { tail -30 "$DEPS/bench_configure.log"; exit 1; }
cmake --build "$SB/build" -j"$JOBS" > "$DEPS/bench_build.log" 2>&1 || { tail -30 "$DEPS/bench_build.log"; exit 1; }
ls "$SB"/build/*_bench "$SB"/build/core_test | sed 's/^/   /'
echo "== self-test (rotation and MatMul gates)"
"$SB/build/core_test" | tail -3
echo "== done. Next:  bash seal_backend/run_seal.sh --tag se_micro --scheme bgv --models irisrf6   (1-minute check)"
