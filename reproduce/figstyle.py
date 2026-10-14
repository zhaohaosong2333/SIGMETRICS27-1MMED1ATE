#!/usr/bin/env python3
"""Single source of truth for every experimental figure in the 1MMED1ATE paper.

One palette, one font size, one width -- this is what makes the paper's figures
look precise (per figure_style.md), not per-figure choices.

Target document: acmart[acmsmall] (ACM POMACS / SIGMETRICS, single column).
  * \\textwidth = \\columnwidth = 395.82pt = 5.478in  (measured from the class)
  * body font = 10pt
So figure_style.md's "single column 3.33in / full 7.0in" (a sigconf two-column
number) does NOT apply here: the one and only column is 5.48in wide. Draw every
figure at its final size and \\includegraphics{fig.pdf} with NO width= -- LaTeX
must never rescale it (rescaling is what desyncs the font sizes across figures).
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── geometry (acmsmall, single column) ──────────────────────────────────
TEXT_W = 5.48          # full single-column text width (in) == \textwidth
COL_W = TEXT_W         # default figure width; draw at this, do not scale
HALF_W = 2.62          # each panel when two figures sit side by side (subfig row)

BODY_PT = 9            # body is 10pt; figures one size smaller, uniform everywhere

# ── Paul Tol muted palette (figure_style.md canonical order) ────────────
TOL = dict(indigo="#332288", cyan="#88CCEE", teal="#44AA99", green="#117733",
           olive="#999933", sand="#DDCC77", rose="#CC6677", wine="#882255",
           purple="#AA4499")

# Solid dark for every spine / tick / label / data edge (no gray, no faded).
INK = "#1A1A1A"
GRID = "#CFCFCF"       # grid is the ONLY thing allowed to be light/thin

# ── semantic color maps (fixed across the whole paper) ──────────────────
# Whenever a *series* is a protocol, it gets this color. Baselines warm, ours teal.
PROTO_ORDER = ["COPSE", "VESTA", "IMM"]
PROTO_COLOR = {"COPSE": TOL["rose"], "VESTA": TOL["olive"],
               "IMM": TOL["teal"], "1MM": TOL["teal"]}
# When color instead encodes the pipeline *stage*, protocols are told apart by hatch.
PROTO_HATCH = {"COPSE": "///", "VESTA": "\\\\\\", "IMM": "", "1MM": ""}

# 3 merged stages for the paper's stacked speed bars (rw / w / p / d).
STAGE3 = ["Compare", "Path Construct", "Encrypt"]
STAGE3_COLOR = {"Compare": TOL["indigo"], "Path Construct": TOL["cyan"],
                "Encrypt": TOL["sand"]}
# 5 raw stages of the collected CSVs.
STAGE5 = ["Compare", "Reorder", "Level Process", "Accumulate", "Encryption"]
STAGE5_COLOR = {"Compare": TOL["indigo"], "Reorder": TOL["cyan"],
                "Level Process": TOL["teal"], "Accumulate": TOL["olive"],
                "Encryption": TOL["sand"]}
# Ordered sequence for "N distinct models/lines" figures (concreteml, etc.).
SEQ = [TOL["indigo"], TOL["teal"], TOL["rose"], TOL["olive"],
       TOL["cyan"], TOL["wine"], TOL["green"], TOL["purple"]]


def use_style():
    """Install the one rcParams profile every figure shares."""
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150,
        "pdf.fonttype": 42, "ps.fonttype": 42,        # TrueType, ACM-safe (no Type-3)
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": BODY_PT, "axes.titlesize": BODY_PT, "axes.labelsize": BODY_PT,
        "xtick.labelsize": BODY_PT, "ytick.labelsize": BODY_PT,
        "legend.fontsize": BODY_PT,
        "text.color": INK, "axes.labelcolor": INK,
        "axes.edgecolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID, "grid.linewidth": 0.5,
        "legend.frameon": False, "legend.handlelength": 1.5,
        "legend.handletextpad": 0.5, "legend.columnspacing": 1.1,
        "legend.labelspacing": 0.35, "legend.borderaxespad": 0.2,
        "hatch.linewidth": 0.5, "hatch.color": INK,
        "lines.linewidth": 1.4, "lines.markersize": 4, "lines.markeredgewidth": 0.0,
        "patch.linewidth": 0.5, "patch.edgecolor": INK,
    })


def new_fig(w=COL_W, h=2.5, **kw):
    """A figure drawn at its exact final size, spacing handled by constrained layout.

    Pair with save() (no bbox_inches='tight') so the PDF page == (w, h) exactly.
    """
    return plt.subplots(figsize=(w, h), layout="constrained", **kw)


def save(fig, path):
    """Write the PDF at exactly figsize -- never tight-crop (that desyncs sizes)."""
    fig.savefig(str(path), format="pdf")
    plt.close(fig)


def bar_kw(color, proto=None):
    """Consistent bar styling: solid dark edge, optional protocol hatch."""
    kw = dict(color=color, edgecolor=INK, linewidth=0.5)
    if proto is not None:
        h = PROTO_HATCH.get(proto, "")
        kw["hatch"] = h or None
    return kw


def line_handles(entries):
    """entries: list of (label, color, marker) -> Line2D legend handles."""
    return [Line2D([0], [0], color=c, marker=m, markersize=4, linewidth=1.4, label=l)
            for l, c, m in entries]
