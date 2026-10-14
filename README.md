# 1MMED1ATE: Private Tree Ensemble Inference via One Homomorphic Matrix-Vector Multiplication

Artifact for the paper *1MMED1ATE: Private Tree Ensemble Inference via One Homomorphic
Matrix-Vector Multiplication* (ACM SIGMETRICS / POMACS). It contains the 1MMED1ATE compiler and
runtime, the re-implementations of the two baselines (COPSE and VESTA) on the same FHE stack, the
trained models, the scripts that ran every measurement of the paper, the measurements themselves,
and the scripts that turn them into the paper's figures and tables.

Everything runs as an unprivileged user: the Python side is one conda environment (OpenFHE comes
as a pip wheel), and the optional SEAL backend builds its own copy of SEAL under `seal_backend/`.
Every experiment can be run either directly on a machine or as Slurm jobs; both paths execute the
same script.

## Contents

| path | what |
|---|---|
| `src/immed1ate/` | compiler (tree model → vectorized representation), runtime (1MM, VESTA, COPSE protocols), OpenFHE BGV/BFV wrappers |
| `scripts/` | experiment runners (speed, memory, sensitivity sweep, motivation), model training/generation, tensor export |
| `data/models/` | the trained models: `rf/`, `gbdt/` (Table 5), `sweep/<family>/` (sensitivity study), `micro/` (smoke tests) |
| `data/train/` | the Income and Soccer training data (Iris ships with scikit-learn, MNIST is downloaded by the trainer) |
| `data/sweep/static/` | hardware-independent profiles of every model (structure, padding, ciphertext counts) |
| `data/paper_results/` | **the paper's measurements**: every result behind the figures and tables, with hardware records |
| `reproduce/` | the experiment harness (`manifest.tsv`, `run_tag.sh`, `run_local.sh`, `slurm/`), the figure and table generators, `make_all.sh` |
| `reproduce/figures_paper/` | the figures and tables regenerated from `data/paper_results/` (identical to the paper's) |
| `seal_backend/` | the C++ port of the three systems to Microsoft SEAL (BGV and BFV) for Figure 15 |
| `tests/` | regression tests of the compiler and of the MVMul kernel |

## Requirements

* Linux x86-64, [conda](https://docs.conda.io) (a fresh Miniconda or Miniforge is enough; the environment is
  built from conda-forge only), internet for the one-time install. No root is needed anywhere.
* Memory (peak resident size, measured; the per-tag table under *Cost of the tags* has every value):
  the smoke tests need under 2 GB; the Category-A tags up to 96 GB (COPSE on the MNIST RF); the sweep
  tags up to 159 GB (`sw_ens_*`, COPSE on the 128-tree forest), 114 GB (`sw_full_*`), 101 GB (`sw_fhe`);
  the SEAL runs up to 105 GB (MNIST). 1MMED1ATE itself never exceeds 45 GB. A run that exceeds the
  machine's memory is killed by the OS and only that part is lost: the speed tags run one process per
  model, the memory and sweep tags one process per (model, system) pair, the collectors keep whatever
  finished, and `run_local.sh` goes on with the next tag. On a 64 GB machine, the pairs above 64 GB are
  COPSE and VESTA on the MNIST RF (Category-A tags and `sw_catA_rf_od`), the largest models of
  `sw_size_*`, `sw_full_*`, `sw_ens_*` and `sw_fhe`, and the SEAL MNIST runs; everything else reproduces.
* The paper's platform: one node with two Intel Xeon Gold 6444Y (2 x 16 cores, 503 GB), 32 OpenMP threads,
  OpenFHE 1.5.1, Python 3.10. Absolute times depend on the machine; the ratios between the systems do not
  (see *Checking the numbers*).

## Setup (5 minutes)

```bash
bash environment/setup_env.sh          # conda env "1mm-core": Python 3.10, openfhe 1.5.1, scikit-learn 1.3.2, ...
conda activate 1mm-core
bash tests/run_tests.sh --quick        # compiler bit-exactness + MVMul replication (~2 min); without --quick ~5 min
bash reproduce/run_local.sh --tags smoke,sw_smoke --omp 4   # both harnesses on tiny models (~2 min)
bash reproduce/make_all.sh             # regenerate every figure/table from the paper's results (~1 min)
```

The last command writes `reproduce/figures_paper/`.

## What reproduces what

| paper item | output file (in the output directory of `make_all.sh`) | generator | experiment tags |
|---|---|---|---|
| Table 5 (model specification) | `tab_modelinfo.md` | `make_extra_tables.py` | none (model files) |
| Table 8 (compiled representation, encrypted-model size) | `tab_compiledrep.md` | `make_compiledrep_table.py` | `sw_*` (ciphertext sizes) |
| Figure 2 (MVMul cost vs depth) | `multiplicativedepth.pdf` | `make_figures.py` | `motiv` |
| Figure 11 and Table 9 (end-to-end time and memory, 8 models) | `rw_ci.pdf`, `tab_absolute_ci.md`, `tab_absolute_paper.tex`, `headline_ci.txt` | `make_end2end.py` | `s_{rf,gbdt}_bgv_{ud,od}`, `m_{rf,gbdt}_bgv_{ud,od}` (3 reps each in the paper) |
| Figure 12 (ensemble size, density, depth, complete trees) | `sweep_scaling.pdf`, `tab_sweep_summary.md`, `sweep_fits.txt` | `make_sweep_figures.py` | `sw_{ens,size,depth,full}_{od,ud}` |
| Precision paragraph | `tab_sweep_precision.md` | `make_sweep_figures.py` | `sw_prec_{od,ud}` |
| Figure 14 (FHE parameters, cost per diagonal product) | `sweep_model.pdf`, `tab_sweep_fheparam.md` | `make_sweep_figures.py` | `sw_fhe` + all `sw_*` |
| Figure 13 (thread scaling) | `threadscaling.pdf` | `make_new_figures.py` | `s_thread` at 1, 2, 4, 8, 16, 32 threads |
| Figure 15 (OpenFHE/SEAL x BGV/BFV) | `backends.pdf` | `make_new_figures.py` | `s_rf_bgv_*`, `s_rf_bfv_*`, `s_rfmnist_bfv_*` + `seal_backend/run_seal.sh` |

The remaining figures of the paper are hand-drawn schematics and the remaining tables are analytical.

## Running the experiments

Every experiment is a *tag* in `reproduce/manifest.tsv` (family, model folder, FHE scheme, depth setting,
cores and time limit). The results of the tags you run land in one directory,
`data/results/<label>/` by default, as `<tag>__<label>__rep<N>.csv` plus `hwlog/` (the hardware that ran
each tag), `logs/`, `sweep_raw/` and `_run/` (scratch). The paper's results are in `data/paper_results/`.

### Plain execution (any machine)

```bash
conda activate 1mm-core
bash reproduce/run_local.sh --all --dry-run                 # list the tags
nohup bash reproduce/run_local.sh --all > run_all.log 2>&1 &   # every tag, one after another
bash reproduce/run_local.sh --tags s_rf_bgv_od,m_rf_bgv_od  # a subset
bash reproduce/run_local.sh --tags s_rf_bgv_od --reps 3     # repetitions (the 95% CIs of Table 9)
```
Threads default to one per physical core (`--omp N` overrides); `--all` also runs the thread-scaling
points up to the core count. Results go to `data/results/<hostname>/`.

### Slurm

```bash
cp reproduce/slurm/site.env.example reproduce/slurm/site.env   # partition, memory, account/mail, env name
srun -p <partition> -c 8 bash tests/run_tests.sh --quick    # the tests use every core: not on the login node
reproduce/slurm/submit.sh --tags smoke,sw_smoke             # one short job per tag
reproduce/slurm/submit.sh --all                             # one job per tag (+ the thread-scaling jobs)
reproduce/slurm/submit.sh --family sweep --reps 3           # repetitions
```
Each job runs `reproduce/slurm/run_one.sbatch`, which only calls `reproduce/run_tag.sh`, the same script
`run_local.sh` uses. Jobs request the manifest's cores and time limit, `--exclusive` (drop it with
`--no-exclusive`), and the partition/memory of `site.env` (or `--partition`, `--mem`). Results go to
`data/results/<partition>/`. Nothing in the harness is specific to a cluster.

### Drawing and checking

```bash
bash reproduce/make_all.sh --results data/results/<label> --outdir out/<label>
```
regenerates every figure and table the results directory has data for. It may be incomplete: each
generator skips what is missing, and Table 9 shows whichever repetitions exist. On other hardware the
latencies and the absolute memory of every system move together; the speedups and memory ratios between
systems are what should match the paper.

### Cost of the tags (paper's platform, 32 threads, one repetition; measured with this harness)

| tags | wall time | peak memory |
|---|---|---|
| `smoke`, `sw_smoke` | under 1 min each | < 2 GB |
| `s_{rf,gbdt}_bgv_{ud,od}` | 32-50 min each (4 tags) | 96 GB (MNIST RF, COPSE) |
| `m_{rf,gbdt}_bgv_{ud,od}` | 32-52 min each (4 tags) | 96 GB |
| `s_rf_bfv_*`, `s_rfmnist_bfv_*` | 12-18 min each (4 tags) | 28 GB (MNIST) |
| `s_thread` x 6 thread counts | 23 min at 1 thread down to 10 min at 8-32 threads | 36 GB |
| `motiv` | 13 min | 9 GB |
| `sw_depth_*` | 22-25 min each | 18 GB |
| `sw_prec_*` | 30-34 min each | 44 GB |
| `sw_size_*` | 50-54 min each | 66 GB |
| `sw_full_*` | 56-62 min each | 114 GB |
| `sw_ens_*` | 1.6-1.7 h each | 159 GB (128 trees, COPSE) |
| `sw_fhe` | 2.7 h | 101 GB |
| `sw_catA_rf_od` | 26 min | 94 GB |
| SEAL: `se_rf` / `se_rfmnist`, BGV | 1.3 h / 1.8 h (single-threaded) | 42 GB / 105 GB |
| SEAL: `se_rf` / `se_rfmnist`, BFV | 3.0 h / 4.3 h (single-threaded) | 42 GB / 105 GB |

All tags together take about 45 node-hours; with `--reps 3` for the eight Category-A tags, about 60.
The sweep tags resume: re-running a tag after a time-out only runs the (model, system) pairs that are
missing from its `sweep_raw/*.jsonl`.

## The SEAL backend (Figure 15)

The portability figure also needs 1MM, VESTA and COPSE in Microsoft SEAL (BGV and BFV, single-threaded
with Intel HEXL). `seal_backend/build.sh` clones SEAL 4.1.2, builds it with HEXL 1.2.5 under
`seal_backend/_deps/` and builds the benches (a few minutes); it needs a C++17 compiler, cmake and git
(`bash environment/setup_env.sh --build-tools` installs them into the conda env if the machine has none).

```bash
bash seal_backend/build.sh                                                   # a few minutes, no root
bash seal_backend/run_seal.sh --tag se_micro --scheme bgv --models irisrf6   # 1-minute check
bash seal_backend/run_seal.sh --tag se_rf --scheme bgv --models "income iris soccer"   # 1.3 h
bash seal_backend/run_seal.sh --tag se_rfmnist --scheme bgv --models mnist             # 1.8 h
#   ... and the same two with --scheme bfv (3 h and 4.3 h); on Slurm: seal_backend/slurm/submit_seal.sh
```
On a cluster whose compute nodes have no internet, fetch on the login node and compile in a job:
`bash seal_backend/build.sh --fetch-only`, then `sbatch -p <partition> -c 16 --wrap "bash seal_backend/build.sh --no-fetch"`,
and submit the runs with `seal_backend/slurm/submit_seal.sh` (same `site.env` as the other jobs).
The runs write `se_<tag>_<scheme>_<OD|UD>__<label>__rep1.csv` into the results directory; together with the
OpenFHE tags `s_rf_bgv_*`, `s_rf_bfv_*` and `s_rfmnist_bfv_*`, `make_all.sh` then draws `backends.pdf`.
Each bench decrypts and checks the winning leaf against the plaintext answer. (As stated in the paper,
VESTA's SEAL-BGV circuit exhausts the noise budget on the MNIST RF; that backend averages the other
three models.)

## Models

All models used by the paper are committed. To retrain or regenerate them:

```bash
python scripts/train_category_a.py          # the RF and GBDT of Income, Soccer, Iris, MNIST (downloads MNIST)
python scripts/gen_sweep_models.py          # the six families of the sensitivity study + data/sweep/static/
```
Both are deterministic with the pinned scikit-learn. The compiler infers the precision from the
thresholds (above 255 means 16 bits), so model file names carry no convention.

## Provenance of the committed results

The committed results were measured on Jarvis, the research-computing cluster of Stevens Institute of
Technology (https://www.stevens.edu/it/services/research-computing). `data/paper_results/hwlog/` holds,
for every committed measurement, the node, CPU, memory, thread count and software versions that executed
it. The Category-A tags ran on the `compute-hf` partition of Jarvis (2 x Xeon Gold 6444Y, HT off); the
memory tags, the OpenFHE-BFV tags and the SEAL runs ran on its `compute-v2` partition, nodes with the same
CPU model. Peak memory and the ratios between systems do not depend on the node; the Category-A latencies
of Table 9 all come from `compute-hf`.

## Citation

```bibtex
@article{zhao2026immed1ate,
  author    = {Zhao, Haosong and Huang, Junhao and Chen, Donglong and Xu, Renjing and Liu, Hongyuan},
  title     = {{1MMED1ATE}: Private Tree Ensemble Inference via One Homomorphic Matrix-Vector Multiplication},
  journal   = {Proceedings of the ACM on Measurement and Analysis of Computing Systems},
  year      = {2026},
  month     = dec,
  volume    = {10},
  number    = {3},
  articleno = {57},
  doi       = {10.1145/3857360},
  publisher = {ACM}
}
```
