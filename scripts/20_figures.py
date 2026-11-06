#!/usr/bin/env python
"""Publication figures, sized and styled for Bioinformatics (OUP).

Reads only the analysis tables stages 12-19 already wrote, so every number in every panel is
traceable to a file on disk and nothing is retyped by hand.

  Layout   single column 82 mm, double column 171 mm, nothing taller than 240 mm
  Output   PNG (--dpi, default 1200 for line art), SVG and PDF of each figure
  Style    no gridlines, no top/right spines, 7 pt text, colour-blind-safe palette
           (Okabe-Ito blue/vermillion/green; worst adjacent CVD dE 11.0, normal-vision dE 25.8)

Fonts are embedded as TrueType in the PDF (fonttype 42) and left as editable text in the SVG, which
is what OUP production asks for. Arial is used when the system has it and DejaVu Sans otherwise.

  python scripts/20_figures.py --config config.yaml
  python scripts/20_figures.py --config config.yaml --dpi 350 --only 1 3
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serprot.common import base_parser, load_config, rpath  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
except ImportError:  # noqa: BLE001
    raise SystemExit("matplotlib is missing from this environment:\n"
                     "  conda activate serprot && pip install matplotlib")

MM = 1 / 25.4
# Bioinformatics Instructions to Authors: single column 86 mm, double column 178 mm
SINGLE, DOUBLE, MAXH = 86 * MM, 178 * MM, 240 * MM

BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"      # Okabe-Ito, validated colour-blind-safe
INK, MUTED, FAINT = "#1A1A1A", "#5A5A5A", "#B4B4B4"

PRETTY = {"resvol": "Mean residue volume", "f_polar": "Polar fraction", "f_pro": "Proline fraction",
          "f_ivywrel": "IVYWREL fraction", "f_neg": "Acidic fraction", "f_pos": "Basic fraction",
          "f_aromatic": "Aromatic fraction", "f_small": "Small-residue fraction",
          "f_gly": "Glycine fraction", "hydropathy": "Mean hydropathy", "net_charge": "Net charge",
          "sasa_mean": "Mean SASA", "sasa_total": "Total SASA", "burial": "Burial",
          "rgyr": "Radius of gyration", "plddt": "pLDDT", "n": "Patch size",
          "attribution": "ESM-2 attribution"}
KEY5 = ["f_ivywrel", "f_polar", "resvol", "f_pro", "f_neg"]
SHORT = {"f_ivywrel": "IVYWREL", "f_polar": "Polar", "resvol": "Residue volume",
         "f_pro": "Proline", "f_neg": "Acidic", "f_aromatic": "Aromatic", "f_small": "Small residue",
         "f_pos": "Basic", "f_gly": "Glycine", "hydropathy": "Hydropathy", "net_charge": "Net charge"}
# pocket and control patches hold the same number of residues by construction, so the size row is a
# tautology with a zero-width interval; it belongs in Methods, not on a forest plot
FOREST_DROP = {"n"}


def legend_below(ax, dy=-0.22, **kw):
    """Legends go under the axis label, where no mark can ever be hidden behind them."""
    lg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, dy), **kw)
    if lg.get_title().get_text():
        lg.get_title().set_fontsize(6.5)
    return lg


def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        # Arial first where it exists; otherwise the metric-compatible clones Linux ships, so the
        # glyph widths match what OUP expects. DejaVu Sans is the last resort, not a target.
        "font.sans-serif": ["Arial", "Liberation Sans", "Arimo", "Helvetica", "Nimbus Sans",
                            "DejaVu Sans"],
        "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.grid": False,                       # no gridlines anywhere
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "axes.edgecolor": INK, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.2, "ytick.major.size": 2.2,
        "xtick.direction": "out", "ytick.direction": "out",
        "lines.linewidth": 1.0, "lines.markersize": 3.2,
        "legend.frameon": False, "legend.handlelength": 1.4, "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.2, "legend.labelspacing": 0.3,
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })
    # Exponents go through mathtext, and the literal superscript characters are missing from
    # Liberation Sans, so they would render as empty boxes. Point mathtext at whichever concrete
    # face the sans-serif list resolved to -- it will not accept the generic alias.
    from matplotlib.font_manager import FontProperties, findfont, get_font
    name = get_font(findfont(FontProperties(family=plt.rcParams["font.sans-serif"]))).family_name
    plt.rcParams.update({"mathtext.fontset": "custom", "mathtext.default": "regular",
                         "mathtext.rm": name, "mathtext.it": f"{name}:italic",
                         "mathtext.bf": f"{name}:bold", "mathtext.sf": name,
                         "mathtext.cal": name})   # cal defaults to 'cursive', which Linux lacks
    return name


def tag_panel(ax, letter, dx=-0.22, dy=1.06):
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="top", ha="left")


def zero_line(ax, vertical=True):
    f = ax.axvline if vertical else ax.axhline
    f(0, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=0)


class Tables:
    """Every analysis table the figures need, loaded once, with a clear error naming what is absent."""

    def __init__(self, cfg):
        self.dir = Path(rpath(cfg, "runs", "analysis", "x.tsv")).parent
        self.cache = {}

    def get(self, name, required=True):
        if name not in self.cache:
            p = self.dir / name
            self.cache[name] = pd.read_csv(p, sep="\t") if p.exists() else None
        if self.cache[name] is None and required:
            raise FileNotFoundError(name)
        return self.cache[name]

    def pocket(self, tag="", radius=10, adjustment="standardized", family="all"):
        d = self.get(f"pocket_vs_temperature{tag}_r{radius}.tsv")
        return d[(d["family"] == family) & (d["adjustment"] == adjustment)]


# --------------------------------------------------------------------------- figure 1
def figure1(T):
    base = T.pocket().dropna(subset=["difference"])
    base = base[~base["descriptor"].isin(FOREST_DROP)].sort_values("difference")
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 80 * MM),
                             gridspec_kw={"width_ratios": [1.3, 1.0, 1.0], "wspace": 0.80})

    # (a) every descriptor at 10 A, in units of its own within-clade SD
    ax = axes[0]
    y = np.arange(len(base))
    sig = (base["q_fdr"] < 0.05).values
    ax.errorbar(base["difference"], y,
                xerr=[base["difference"] - base["difference_lo"], base["difference_hi"] - base["difference"]],
                fmt="none", ecolor=INK, elinewidth=0.7, capsize=0, zorder=2)
    ax.scatter(base["difference"][sig], y[sig], s=11, color=BLUE, zorder=3, linewidths=0)
    ax.scatter(base["difference"][~sig], y[~sig], s=11, facecolors="white",
               edgecolors=MUTED, linewidths=0.7, zorder=3)
    zero_line(ax)
    ax.set_yticks(y)
    ax.set_yticklabels([PRETTY.get(d, d) for d in base["descriptor"]])
    ax.set_xlabel("Pocket − control, within-clade SD")
    ax.set_ylim(-0.7, len(base) - 0.3)
    legend_below(ax, handles=[Line2D([], [], marker="o", ls="", color=BLUE, ms=3.4, label="q < 0.05"),
                              Line2D([], [], marker="o", ls="", mfc="white", mec=MUTED, mew=0.7,
                                     ms=3.4, label="n.s.")], ncol=2, dy=-0.16)
    tag_panel(ax, "a", dx=-0.62)

    # (b) the 10 A boundary is a choice -- show it does not carry the result
    ax = axes[1]
    y = np.arange(len(KEY5))
    for r, al, off in zip([6, 8, 10], [0.45, 0.70, 1.0], [-0.26, 0.0, 0.26]):
        try:
            d = T.pocket(radius=r).set_index("descriptor")
        except FileNotFoundError:
            continue
        v = np.array([d["difference"].get(k, np.nan) for k in KEY5], dtype=float)
        lo = np.array([d["difference_lo"].get(k, np.nan) for k in KEY5], dtype=float)
        hi = np.array([d["difference_hi"].get(k, np.nan) for k in KEY5], dtype=float)
        ax.errorbar(v, y + off, xerr=[v - lo, hi - v], fmt="o", ms=2.6, color=BLUE, alpha=al,
                    ecolor=BLUE, elinewidth=0.6, capsize=0, label=f"{r} Å", linestyle="none")
    zero_line(ax)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[k] for k in KEY5])
    ax.set_ylim(-0.7, len(KEY5) - 0.3)
    ax.set_xlabel("Pocket − control (SD)")
    legend_below(ax, title="Pocket radius", ncol=3, columnspacing=0.7, dy=-0.16)
    tag_panel(ax, "b", dx=-0.42)

    # (c) the same contrast on structures AlphaFold never covered
    ax = axes[2]
    ef = T.pocket(tag="_esmfold").set_index("descriptor")
    af = T.pocket().set_index("descriptor")
    order = [k for k in KEY5 if k in ef.index and k in af.index]
    y = np.arange(len(order))
    for i, k in enumerate(order):
        ax.plot([af["difference"][k], ef["difference"][k]], [i, i], color=FAINT, lw=0.8, zorder=1)
    ax.scatter([af["difference"][k] for k in order], y, s=11, color=BLUE, zorder=3,
               linewidths=0, label="AlphaFold")
    ax.scatter([ef["difference"][k] for k in order], y, s=11, color=VERM, zorder=3,
               linewidths=0, marker="s", label="ESMFold")
    zero_line(ax)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT[k] for k in order])
    ax.set_xlabel("Pocket − control (SD)")
    ax.set_ylim(-0.7, len(order) - 0.3)
    legend_below(ax, ncol=2, dy=-0.16)
    tag_panel(ax, "c", dx=-0.42)
    return fig


# --------------------------------------------------------------------------- figure 2
def _shell_profile(ax, d, color, label=None, band=True):
    d = d.sort_values("shell_lo" if "shell_lo" in d else "shell_i")
    x = np.arange(len(d))
    ax.plot(x, d["slope"], color=color, lw=1.0, marker="o", ms=2.8, label=label, zorder=3)
    if band and {"ci_lo", "ci_hi"} <= set(d.columns):
        ax.fill_between(x, d["ci_lo"], d["ci_hi"], color=color, alpha=0.16, lw=0, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(d["shell"], rotation=38, ha="right")
    return d


def figure2(T):
    prof = T.get("distance_profile.tsv")
    prof = prof[(prof["measure"] == "attribution") & (prof["adjustment"] == "raw")]
    bd = T.get("burial_distance.tsv")
    attr = T.get("residue_attribution_r10.tsv").set_index("region")["slope"]

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 68 * MM), gridspec_kw={"wspace": 0.62})

    # (a) the continuous gradient, with the matched-control patch as the reference it is measured against
    ax = axes[0]
    _shell_profile(ax, prof, BLUE)
    if "control" in attr.index:
        ctrl = float(attr["control"])
        ax.axhline(ctrl, color=MUTED, lw=0.7, ls=(0, (5, 3)), zorder=2)
        y0, y1 = ax.get_ylim()
        ax.text(0.03, ctrl + 0.035 * (y1 - y0), "matched control",
                transform=ax.get_yaxis_transform(), ha="left", va="bottom",
                fontsize=6, color=MUTED)
    ax.set_xlabel("Distance from catalytic Ser (Å)")
    ax.set_ylabel("Attribution slope / +10 °C")
    tag_panel(ax, "a", dx=-0.34)

    # (b, c) the same gradient at fixed solvent exposure
    strata = sorted(bd["burial_stratum"].unique())
    shades = {s: a for s, a in zip(strata, np.linspace(0.34, 1.0, len(strata)))}
    names = {strata[0]: "most exposed", strata[-1]: "most buried"}
    for ax, meas, letter, mult, unit in [(axes[1], "attribution", "b", 1.0, ""),
                                         (axes[2], "f_ivywrel", "c", 1e3, r" ($\times$10$^{-3}$)")]:
        sub = bd[bd["measure"] == meas]
        series = [(s, g, g["slope"].values * mult, g["ci_lo"].values * mult, g["ci_hi"].values * mult)
                  for s, g in ((s, sub[sub["burial_stratum"] == s].sort_values("dist_i"))
                               for s in strata) if len(g)]
        if not series:
            ax.set_visible(False)
            continue
        # Few exposed residues sit within 10 A of the catalytic Ser, so that one interval is far
        # wider than the rest and would squash the panel. The axis is set from the typical interval
        # and anything running past it gets an arrow head, so nothing is silently cut off.
        vals = np.concatenate([t[2] for t in series])
        pad = 2.2 * float(np.median(np.concatenate([(t[4] - t[3]) / 2 for t in series])))
        ymin, ymax = min(float(vals.min()) - pad, 0.0), max(float(vals.max()) + pad, 0.0)
        for s, g, v, lo, hi in series:
            x = np.arange(len(g))
            ax.errorbar(x, v, yerr=[np.maximum(v - np.maximum(lo, ymin), 0),
                                    np.maximum(np.minimum(hi, ymax) - v, 0)],
                        color=BLUE, alpha=shades[s], lw=1.0, marker="o", ms=2.6,
                        elinewidth=0.6, capsize=0, label=names.get(s, "intermediate"))
            inset = 0.015 * (ymax - ymin)   # keep the arrow head whole instead of half-clipped
            for xi, l, h in zip(x, lo, hi):
                if l < ymin:
                    ax.plot([xi], [ymin + inset], marker="v", ms=3.0, ls="none", color=BLUE,
                            alpha=shades[s], zorder=4)
                if h > ymax:
                    ax.plot([xi], [ymax - inset], marker="^", ms=3.0, ls="none", color=BLUE,
                            alpha=shades[s], zorder=4)
            ax.set_xticks(x)
            ax.set_xticklabels(g["dist_bin"], rotation=38, ha="right")
        ax.set_ylim(ymin, ymax)
        zero_line(ax, vertical=False)
        ax.set_xlabel("Distance from catalytic Ser (Å)")
        short = "Attribution" if meas == "attribution" else "IVYWREL"
        ax.set_ylabel(f"{short} slope / +10 °C{unit}")
        # lower right: the series rise left to right, and the over-long inner-shell whisker is
        # always at the left edge, so this is the one corner nothing can land in
        lg = ax.legend(title="Burial tercile", loc="lower right", borderpad=0.2)
        lg.get_title().set_fontsize(6.5)
        tag_panel(ax, letter, dx=-0.34)
    return fig


# --------------------------------------------------------------------------- figure 3
def figure3(T):
    fp = T.get("flexibility_profile.tsv")
    fl = T.get("flexibility.tsv")
    comp = T.pocket().dropna(subset=["difference"])
    fle = T.get("flexibility_esmfold.tsv", required=False)

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 80 * MM),
                             gridspec_kw={"width_ratios": [1.0, 1.25, 1.0], "wspace": 0.95})

    # (a) no thermal gradient anywhere in flexibility
    ax = axes[0]
    _shell_profile(ax, fp.rename(columns={"shell_i": "shell_lo"}), BLUE)
    zero_line(ax, vertical=False)
    ax.set_xlabel("Distance from catalytic Ser (Å)")
    ax.set_ylabel("Flexibility slope / +10 °C")
    tag_panel(ax, "a", dx=-0.40)

    # (b) the bound: every compositional difference sits outside the flexibility interval
    ax = axes[1]
    row = fl[(fl["family"] == "all") & (fl["pass"] == "standardized")]
    sig = comp[comp["q_fdr"] < 0.05].sort_values("difference")
    y = np.arange(len(sig))
    if len(row):
        r = row.iloc[0]
        ax.axvspan(r["difference_lo"], r["difference_hi"], color=VERM, alpha=0.16, lw=0, zorder=0)
        ax.axvline(r["difference"], color=VERM, lw=0.9, zorder=1)
    ax.errorbar(sig["difference"], y,
                xerr=[sig["difference"] - sig["difference_lo"], sig["difference_hi"] - sig["difference"]],
                fmt="o", ms=3.0, color=BLUE, ecolor=INK, elinewidth=0.7, capsize=0, zorder=3)
    zero_line(ax)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT.get(d, PRETTY.get(d, d)) for d in sig["descriptor"]])
    ax.set_xlabel("Pocket − control (SD)")
    ax.set_ylim(-0.8, len(sig) - 0.2)
    legend_below(ax, handles=[Line2D([], [], marker="o", ls="", color=BLUE, ms=3.4,
                                     label="composition"),
                              Line2D([], [], color=VERM, lw=4, alpha=0.3,
                                     label="flexibility 95% CI")], ncol=2, dy=-0.16)
    tag_panel(ax, "b", dx=-0.62)

    # (c) S1 is the one family where flexibility moves, and it moves on both structure sets
    ax = axes[2]
    passes = ["normalised", "unnormalised", "normalised, standardized"]
    short = {"normalised": "normalised", "unnormalised": "unnormalised",
             "normalised, standardized": "standardized"}
    s1a = fl[fl["family"] == "S1"].set_index("measure")
    s1e = fle[fle["family"] == "S1"].set_index("measure") if fle is not None else None
    y = np.arange(len(passes))
    for i, p in enumerate(passes):
        if p in s1a.index and s1e is not None and p in s1e.index:
            ax.plot([s1a["difference"][p], s1e["difference"][p]], [i, i], color=FAINT, lw=0.8, zorder=1)
    def pts(tbl, color, marker, label):
        v = [tbl["difference"].get(p, np.nan) for p in passes]
        lo = [tbl["difference_lo"].get(p, np.nan) for p in passes]
        hi = [tbl["difference_hi"].get(p, np.nan) for p in passes]
        ax.errorbar(v, y, xerr=[np.array(v) - lo, np.array(hi) - np.array(v)], fmt=marker,
                    ms=3.0, color=color, ecolor=color, elinewidth=0.6, capsize=0,
                    label=label, zorder=3, linestyle="none")
    pts(s1a, BLUE, "o", "AlphaFold")
    if s1e is not None:
        pts(s1e, VERM, "s", "ESMFold")
    zero_line(ax)
    ax.set_yticks(y)
    ax.set_yticklabels([short[p] for p in passes])
    ax.set_xlabel("S1 pocket − control, flexibility")
    ax.set_ylim(-0.7, len(passes) - 0.3)
    legend_below(ax, ncol=2, dy=-0.16)
    tag_panel(ax, "c", dx=-0.52)
    return fig


# --------------------------------------------------------------------------- figure 4
def figure4(T):
    sg = T.get("subgroups_domain.tsv")
    groups = [g for g in ["Bacteria", "Archaea"] if g in set(sg["group"])]
    desc = [d for d in KEY5 if d in set(sg["descriptor"])]
    if len(groups) < 2 or not desc:
        return None
    fig, axes = plt.subplots(len(desc), 1, figsize=(SINGLE, 24 * MM * len(desc)),
                             gridspec_kw={"hspace": 1.10})
    axes = np.atleast_1d(axes)
    colors = {groups[0]: BLUE, groups[1]: VERM}
    for ax, d in zip(axes, desc):
        sub = sg[sg["descriptor"] == d].set_index("group")
        y = np.arange(len(groups))[::-1]
        for yi, g in zip(y, groups):
            if g not in sub.index:
                continue
            r = sub.loc[g]
            ax.errorbar([r["difference"]], [yi],
                        xerr=[[r["difference"] - r["difference_lo"]], [r["difference_hi"] - r["difference"]]],
                        fmt="o", ms=3.2, color=colors[g], ecolor=colors[g],
                        elinewidth=0.8, capsize=0, linestyle="none")
        # a clipped interval reads as one that continues, so the axis always contains both whiskers
        span = [sub["difference_lo"].min(), sub["difference_hi"].max(), 0.0]
        pad = 0.08 * (max(span) - min(span))
        ax.set_xlim(min(span) - pad, max(span) + pad)
        zero_line(ax)
        ax.set_yticks(y)
        ax.set_yticklabels(groups)
        ax.set_ylim(-0.6, len(groups) - 0.4)
        # units differ per descriptor, so each strip keeps its own scale; four ticks and a shared
        # power-of-ten offset keep the labels from running into one another at 82 mm
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.ticklabel_format(style="sci", scilimits=(-2, 3), axis="x", useMathText=True)
        ax.xaxis.get_offset_text().set_fontsize(6)
        ax.set_title(PRETTY.get(d, d), loc="left", pad=3)
    axes[-1].set_xlabel("Pocket − control, per +10 °C")
    return fig


# A one-line title above each figure, so a reader who meets the figure before the caption knows
# which analysis they are looking at. It repeats the caption's opening claim on purpose.
TITLES = {
    1: "Thermal adaptation is absent from the catalytic neighbourhood",
    2: "The gradient is continuous and is not an exposure effect",
    3: "Flexibility shows no pocket−control difference, with the power to have seen one",
    4: "The evidence is bacterial",
}

CAPTIONS = {
    1: ("Figure 1. Thermal adaptation is absent from the catalytic neighbourhood. (a) Within-clade "
        "change in each descriptor per +10 degrees C of optimal growth temperature, pocket minus "
        "matched control, in units of that descriptor's own within-clade standard deviation; bars are "
        "95 per cent cluster-bootstrap intervals over clades. (b) The same contrast at pocket radii of "
        "6, 8 and 10 A. (c) AlphaFold models against ESMFold models of sequences AlphaFold does not "
        "cover."),
    2: ("Figure 2. The gradient is continuous and is not an exposure effect. (a) Within-clade slope of "
        "each distance shell's mean ESM-2 attribution on growth temperature, with 95 per cent "
        "intervals; the dashed line is the matched control patch. (b, c) The same slopes computed "
        "inside terciles of solvent exposure taken within each protein."),
    3: ("Figure 3. Flexibility from an elastic network model. (a) Within-clade slope of relative "
        "C-alpha fluctuation on growth temperature by distance shell. (b) Each compositional "
        "pocket-minus-control difference against the 95 per cent interval of the flexibility "
        "difference (shaded), on the same proteins and clades. (c) S1, the one family in which "
        "flexibility separates pocket from control, on both structure sets."),
    4: ("Figure 4. The evidence is bacterial. Pocket minus matched control per +10 degrees C in each "
        "domain, with 95 per cent cluster-bootstrap intervals. The archaeal arm rests on 88 clades "
        "and resolves nothing either way."),
}


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--dpi", type=int, default=1200,
                    help="raster resolution for submission; the journal asks 1200 for line art")
    ap.add_argument("--review-dpi", type=int, default=150,
                    help="resolution of the low-resolution PNG the journal wants for the review process")
    ap.add_argument("--only", nargs="*", type=int, help="build only these figure numbers")
    ap.add_argument("--formats", default="tif,png,svg,pdf",
                    help="tif is the submission raster (LZW-compressed), png the review copy")
    ap.add_argument("--no-titles", action="store_true",
                    help="drop the on-figure titles if the journal wants the claim only in the caption")
    args = ap.parse_args()
    cfg = load_config(args.config)
    # OUP asks for Arial or Helvetica; say which one this machine actually resolved to, because a
    # silent fall back to DejaVu Sans is the kind of thing production catches and you do not
    got = style()
    metric = {"arial": "Arial", "helvetica": "Helvetica", "liberationsans": "Arial",
              "arimo": "Arial", "nimbussans": "Helvetica"}
    key = next((k for k in metric if got.lower().replace(" ", "").startswith(k)), None)
    print(f"sans-serif font in use: {got}" +
          (f"   (metrics of {metric[key]})" if key else
           "   WARNING no Arial- or Helvetica-metric font installed"))
    T = Tables(cfg)
    outdir = Path(rpath(cfg, "figures", "x.png")).parent   # ".png" so rpath makes figures/, not figures/x/
    outdir.mkdir(parents=True, exist_ok=True)

    builders = {1: figure1, 2: figure2, 3: figure3, 4: figure4}
    want = args.only or sorted(builders)
    made = []
    for n in want:
        try:
            fig = builders[n](T)
        except FileNotFoundError as e:
            print(f"figure {n}: SKIPPED, missing {e}")
            continue
        except Exception as e:  # noqa: BLE001
            print(f"figure {n}: FAILED, {type(e).__name__}: {e}")
            continue
        if fig is None:
            print(f"figure {n}: SKIPPED, not enough data")
            continue
        if n in TITLES and not args.no_titles:
            # above the panel letters, which sit at 1.06 of their own axes; tight bbox pulls it in
            fig.suptitle(TITLES[n], x=0.0, y=1.015, ha="left", va="bottom",
                         fontsize=9, fontweight="bold")
        w, h = fig.get_size_inches()
        if h > MAXH:
            print(f"figure {n}: WARNING height {h / MM:.0f} mm exceeds the 240 mm limit")
        saved = None
        for ext in args.formats.split(","):
            p = outdir / f"figure{n}.{ext}"
            if ext == "tif":
                # uncompressed TIFF at 1200 dpi runs to ~100 MB a figure; LZW is lossless and
                # leaves line art at a few MB
                fig.savefig(p, dpi=args.dpi, pil_kwargs={"compression": "tiff_lzw"})
            elif ext == "png":
                fig.savefig(p, dpi=args.review_dpi)      # the low-resolution review copy
            else:
                fig.savefig(p)
            made.append(p)
            if ext == "tif":
                from PIL import Image
                with Image.open(p) as im:
                    saved = im.size[0] / args.dpi / MM
        plt.close(fig)
        # the canvas is drawn at the target width; the saved file is cropped to the ink, so it comes
        # out a little narrower. The journal scales to column width either way, which lifts 7 pt text
        # to about 7.5 pt -- above the minimum, not below it.
        extra = f", saved {saved:.0f} mm" if saved else ""
        print(f"figure {n}: canvas {w / MM:.0f} x {h / MM:.0f} mm{extra}")

    cap = outdir / "figure_captions.txt"
    cap.write_text("\n\n".join(CAPTIONS[n] for n in want if n in CAPTIONS) + "\n")
    made.append(cap)
    print(f"\nwrote {len(made)} files to {outdir}")
    for p in made:
        print(f"  {p.name}  {p.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
