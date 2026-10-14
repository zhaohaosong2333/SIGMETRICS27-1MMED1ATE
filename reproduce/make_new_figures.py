#!/usr/bin/env python3
"""Generate the two cross-cutting paper figures that need multi-backend / multi-thread data.

Both land in figures_paper/ with the paper's own filenames (like make_figures.py):

  backends.pdf       [Figure 15]     1MM/VESTA/COPSE within each of FOUR backends -- OpenFHE and
                                        SEAL, each in BGV and BFV -- normalized to that backend's own
                                        COPSE, on the real RF models (income/iris/soccer/mnist, geo-mean).
                                        A within-backend portability figure: cross-backend heights are
                                        NOT head-to-head (SEAL is single-thread+HEXL, OpenFHE 32t).
  threadscaling.pdf  [Figure 13] parallel speedup vs OMP threads {1..32} (UD), Income RF.

Shares figstyle.py (one Paul Tol palette, 9pt, single-column width). Each figure skips gracefully
if its data is not present yet.
    python reproduce/make_new_figures.py [--results data/paper_results] [--outdir reproduce/figures_paper]
"""
import os, csv, glob, argparse, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))   # for figstyle
import figstyle as fs

HJ = "data/paper_results"   # results directory; set by --results


def _label_of(tag):
    """The provenance label of a tag's files in the results directory (<tag>__<label>__rep1*.csv)."""
    g = sorted(glob.glob(f"{HJ}/{tag}__*__rep1*.csv"))
    return os.path.basename(g[0]).split("__")[1] if g else None


def load_speed(path):
    """*Result.csv -> {model: {proto: [5 stage ms]}}."""
    d = {}
    if not os.path.exists(path):
        return d
    with open(path) as f:
        r = csv.reader(f); next(r)
        for row in r:
            if not row or not row[0]:
                continue
            proto, model = row[0].split("-", 1)
            d.setdefault(model, {})[proto] = [float(x) for x in row[1:6]]
    return d


def tot(stages):
    return sum(stages)


def fig_thread_scaling(out):
    # the Income RF at UD (tag s_thread, one file per thread count: __omp<N>); all protocols share the depth
    pts, model = [], None
    for tag, m in [("s_thread", "income")]:
        pts = []
        part = _label_of(tag)
        if part is None:
            continue
        for omp in [1, 2, 4, 8, 16, 32]:
            cands = [f"{HJ}/{tag}__{part}__rep1__omp{omp}.csv"]
            if omp == 32:
                cands.append(f"{HJ}/{tag}__{part}__rep1.csv")  # old sweep: 32 = base rep1
            d = {}
            for c in cands:
                if os.path.exists(c):
                    d = load_speed(c); break
            if d and m in d:
                pts.append((omp, d))
        if len(pts) >= 3:
            model = m; break
    if len(pts) < 3:
        print("  [skip] thread_scaling: <3 thread points"); return
    threads = [o for o, _ in pts]
    # narrow (half-column) so it can sit in a paper-side wrapfigure with native ~9pt labels
    # instead of shrinking under a rescale; short enough to fit beside the Parallel-Scalability para.
    fig, ax = fs.new_fig(w=fs.HALF_W, h=1.9)
    # Normalize away absolute time: plot each protocol's parallel speedup t(1)/t(n) -- a strong-scaling
    # view. The dashed diagonal is perfect linear scaling; real FHE inference peels away almost at once
    # (memory-bound), all three protocols saturate ~16 threads and regress at 32 (cross-NUMA on 2 sockets).
    ax.plot(threads, threads, ls="--", lw=0.9, color=fs.INK, label="Ideal (linear)", zorder=1)
    for pr, key in [("IMM", "1MM"), ("VESTA", "VESTA"), ("COPSE", "COPSE")]:
        series = [(o, tot(d[model][key])) for o, d in pts if model in d and key in d[model]]
        if not series:
            continue
        t1 = series[0][1]                                  # this protocol's own 1-thread time
        xs = [o for o, _ in series]
        ys = [t1 / v for _, v in series]                   # parallel speedup vs 1 thread
        ax.plot(xs, ys, "-o", color=fs.PROTO_COLOR[pr], markersize=3.5,
                label=("1MM" if pr == "IMM" else pr), zorder=3)
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xticks(threads); ax.set_xticklabels(threads)
    ax.set_yticks([1, 2, 4]); ax.set_yticklabels([1, 2, 4])
    ax.set_xlim(0.85, 40); ax.set_ylim(0.85, 4.4)
    ax.set_xlabel("OMP threads")
    ax.set_ylabel("Speedup vs 1 thread ($\\times$)")
    ax.grid(True, which="both", ls=":", lw=0.5, color=fs.GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")   # UD / model noted in the caption, so no legend title
    fs.save(fig, out / "threadscaling.pdf")
    print("  saved threadscaling.pdf")


def _load_speed_multi(tags, cfg):
    """Merge *speed CSVs for a backend: each of `tags` at config `cfg` (try lower/UPPER filename case,
    any label). Returns {model: {proto: [5 stage ms]}}. Lets SEAL's rf runs combine."""
    merged = {}
    for tag in tags:
        for c in (cfg, cfg.upper()):
            # each backend in its OWN standard config: OpenFHE at the paper's 32 threads (rep1),
            # SEAL single-thread + HEXL (rep1). This is a portability (within-backend) figure, so
            # cross-backend magnitudes are not head-to-head; see caption.
            g = sorted(glob.glob(f"{HJ}/{tag}_{c}__*__rep1.csv"))
            found = g[0] if g else None
            if found:
                for m, v in load_speed(found).items():
                    merged.setdefault(m, {}).update(v)
                break
    return merged


def fig_backends(out):
    """1MM's advantage is PORTABLE across FHE schemes and libraries: within each backend -- run in its
    OWN standard configuration (SEAL single-thread + HEXL; OpenFHE at the paper's 32 threads) -- 1MM's
    runtime is a small fraction of BOTH its VESTA and COPSE baselines, at Optimized (OD) and Unified (UD)
    depth. Bars are per-model time normalized to that backend's own COPSE (geo-mean over the real RF
    models income/iris/soccer/mnist); the labels on the 1MM bars are its speedup over COPSE. Each group is a
    WITHIN-backend comparison; cross-backend bar heights are not head-to-head. Four series are told apart
    by hatch as well as color (colorblind-safe): COPSE ///, VESTA \\\\, 1MM-UD dots, 1MM-OD xxx."""
    REAL = ["income", "iris", "soccer", "mnist"]
    # The MNIST RF is its own tag in OpenFHE-BFV and SEAL (s_rfmnist_*, se_rfmnist_*); later tags
    # override earlier ones per model.
    BACKENDS = [("OpenFHE\nBGV", ["s_rf_bgv"]),
                ("OpenFHE\nBFV", ["s_rf_bfv", "s_rfmnist_bfv"]),
                ("SEAL\nBGV",    ["se_rf_bgv", "se_rfmnist_bgv"]),
                ("SEAL\nBFV",    ["se_rf_bfv", "se_rfmnist_bfv"])]

    # (backend, model) pairs left out of a backend's geo-mean because a protocol's decrypted result failed
    # the golden check (the SEAL benches still write their timings). SEAL-BGV VESTA on MNIST exhausts the
    # noise budget even at the largest 128-bit modulus for N = 2^15 (noise budget 0 at 840 and at 881 bits;
    # Income keeps 21 bits).
    INCORRECT = {("SEAL\nBGV", "mnist")}

    def gmean(xs):
        return float(np.exp(np.mean(np.log(xs)))) if xs else float("nan")

    data = []   # (label, COPSE=1, VESTA/COPSE, 1MM-OD/COPSE, 1MM-UD/COPSE)
    for lab, tags in BACKENDS:
        od, ud = _load_speed_multi(tags, "od"), _load_speed_multi(tags, "ud")
        ms = [m for m in REAL if (lab, m) not in INCORRECT
              and m in od and all(p in od[m] for p in ("1MM", "VESTA", "COPSE"))]
        print(f"    {lab.replace(chr(10), '-')}: geo-mean over {', '.join(ms)}")
        if not ms:
            print(f"  [backends] missing data for {lab.replace(chr(10), ' ')}"); continue
        vesta = gmean([tot(od[m]["VESTA"]) / tot(od[m]["COPSE"]) for m in ms])
        immod = gmean([tot(od[m]["1MM"])   / tot(od[m]["COPSE"]) for m in ms])
        msu = [m for m in ms if m in ud and "1MM" in ud[m]]
        immud = gmean([tot(ud[m]["1MM"]) / tot(od[m]["COPSE"]) for m in msu]) if msu else float("nan")
        data.append((lab, 1.0, vesta, immod, immud))
    if not data:
        print("  [skip] backends: no backend data present"); return

    from matplotlib.patches import Patch
    fig, ax = fs.new_fig(w=fs.COL_W, h=2.0)
    x = np.arange(len(data)); w = 0.2
    # distinct hatches so the four series read without color (colorblind-safe)
    series = [("COPSE", 1, fs.PROTO_COLOR["COPSE"], "///"),
              ("VESTA", 2, fs.PROTO_COLOR["VESTA"], "\\\\\\"),
              ("1MM (UD)", 4, fs.TOL["cyan"], "..."),
              ("1MM (OD)", 3, fs.PROTO_COLOR["IMM"], "xxx")]
    for k, (_, idx, col, h) in enumerate(series):
        ax.bar(x + (k - 1.5) * w, [d[idx] for d in data], w, hatch=h, **fs.bar_kw(col))
    # label the SPEEDUP over COPSE (= 1 / normalized height) above BOTH 1MM bars
    # (UD is the 3rd bar at +0.5w, OD the 4th at +1.5w -- matching the series order above)
    for xi, d in zip(x, data):
        if d[4] == d[4]:
            ax.text(xi + 0.5 * w, d[4] + 0.015, f"{1 / d[4]:.1f}$\\times$", ha="center", va="bottom",
                    fontsize=fs.BODY_PT - 2, color=fs.INK)
        ax.text(xi + 1.5 * w, d[3] + 0.015, f"{1 / d[3]:.1f}$\\times$", ha="center", va="bottom",
                fontsize=fs.BODY_PT - 2, color=fs.INK)
    ax.axhline(1.0, ls=":", lw=0.7, color=fs.GRID); ax.set_axisbelow(True)   # COPSE baseline
    ax.set_xticks(x); ax.set_xticklabels([d[0] for d in data])
    ax.set_ylabel("Runtime / COPSE")
    ax.set_ylim(0, 1.35)   # headroom so the in-axes legend clears the COPSE bars (at 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])   # data only reaches 1.0; no ticks in the legend band
    # legend lowered INTO the axes' empty top band (no external strip) -> shorter figure
    ax.legend(handles=[Patch(facecolor=c, edgecolor=fs.INK, linewidth=0.5, hatch=h, label=l)
                       for l, _, c, h in series],
              loc="upper center", ncol=4, frameon=False, columnspacing=1.3,
              handlelength=1.3, handletextpad=0.5, borderaxespad=0.2)
    fs.save(fig, out / "backends.pdf")
    print(f"  saved backends.pdf  ({len(data)} backends; runtime/COPSE; "
          f"1MM-OD speedup: {['%.1f' % (1 / d[3]) for d in data]})")
    for lab, _, vesta, immod, immud in data:      # the numbers the Figure 15 paragraph quotes
        print(f"    {lab.replace(chr(10), '-'):13s} VESTA/COPSE={vesta:.2f}  1MM-OD speedup={1 / immod:.2f}x  "
              f"1MM-UD speedup={1 / immud:.2f}x  1MM vs VESTA (OD)={vesta / immod:.2f}x")


def main():
    global HJ
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=HJ, help="results directory (default data/paper_results)")
    ap.add_argument("--outdir", default="reproduce/figures_paper")
    args = ap.parse_args()
    HJ = args.results
    fs.use_style()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    print(f"paper figures -> {out}")
    for fn in (fig_backends, fig_thread_scaling):
        try:
            fn(out)
        except Exception as e:
            print(f"  [warn] {fn.__name__}: {e}")
    print("Done.")


if __name__ == "__main__":
    main()
