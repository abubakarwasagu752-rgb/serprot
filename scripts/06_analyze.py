#!/usr/bin/env python
"""Evaluate all runs and test the stability-specificity trade-off.

(A) performance   held-out OGT and breadth accuracy for every model/split (lineage-aware folds)
(A2) leakage      per fold: can z_S predict OGT, can z_T predict breadth?  dCor(z_T, z_S)
(B) observed      within-clade (cluster fixed-effects) slope of observed breadth on observed OGT
(C) predicted     the same slope on held-out model predictions for ALL proteins (labeled or not), cluster-bootstrap CI

Reading guide: a negative slope = hotter relatives have broader/narrower specificity? Breadth is 0 (strict) .. 1 (flat),
so a NEGATIVE slope means thermophilic relatives are more specific (narrower), i.e. a trade-off in the predicted direction.
The slope is measured WITHIN clades of relatives (cluster_pairs), which removes lineage as a confounder.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serprot.common import base_parser, load_config, load_dataset, rpath, safe_corr, seed_all  # noqa: E402

# colours: validated default categorical slots 1-3 (first three pass the all-pairs checks), ink tokens
FAM_COLORS = {"S8": "#2a78d6", "S9": "#eb6834", "S1": "#1baf7a"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2de", "#fcfcfb"
BAR, BAR_REF = "#2a78d6", "#a9a8a2"


def dcor2(x, y):
    def c(a):
        d = np.sqrt(((a[:, None] - a[None]) ** 2).sum(-1))
        return d - d.mean(0) - d.mean(1)[:, None] + d.mean()
    A, B = c(x), c(y)
    return float(max((A * B).mean(), 0) / np.sqrt((A * A).mean() * (B * B).mean() + 1e-12))


def fe_slope(d, x, y, cl, n_boot=2000, seed=0, scale=10.0):
    """Cluster fixed-effects slope of y on x (per `scale` units of x) with a cluster-bootstrap CI."""
    d = d.dropna(subset=[x, y, cl])
    d = d[d.groupby(cl)[x].transform("size") >= 2]
    if len(d) < 6:
        return dict(slope=np.nan, lo=np.nan, hi=np.nan, p=np.nan, n=len(d), clusters=0)
    dx = d[x] - d.groupby(cl)[x].transform("mean")
    dy = d[y] - d.groupby(cl)[y].transform("mean")
    st = pd.DataFrame({"c": d[cl].values, "sxy": (dx * dy).values, "sxx": (dx * dx).values}).groupby("c").sum()
    sxy, sxx = st["sxy"].values, st["sxx"].values
    if sxx.sum() <= 0:
        return dict(slope=np.nan, lo=np.nan, hi=np.nan, p=np.nan, n=len(d), clusters=len(st))
    est = sxy.sum() / sxx.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(st), size=(n_boot, len(st)))
    bs = sxy[idx].sum(1) / np.maximum(sxx[idx].sum(1), 1e-12)
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return dict(slope=est * scale, lo=np.percentile(bs, 2.5) * scale, hi=np.percentile(bs, 97.5) * scale,
                p=min(p, 1.0), n=len(d), clusters=len(st))


def load_runs(cfg, ds):
    runs = rpath(cfg, "runs")
    out = {}
    for d in sorted(runs.iterdir()):
        if not d.is_dir() or d.name == "analysis":
            continue
        model, split = d.name.rsplit("_", 1)
        if (d / "preds.tsv").exists():
            p = pd.read_csv(d / "preds.tsv", sep="\t")
        else:
            parts = [pd.read_csv(f / "preds.tsv", sep="\t") for f in sorted(d.glob("fold*")) if (f / "preds.tsv").exists()]
            if not parts:
                continue
            p = pd.concat(parts)
        cols = ["accession", "family", "cluster_split", "cluster_pairs", "ogt", "breadth", "spec_n", "phylum"]
        out[(model, split)] = (d, p.merge(ds[cols], on="accession"))
    return out


def performance(runs):
    rows = []
    for (model, split), (_, p) in runs.items():
        for fam in ["all"] + sorted(p["family"].unique()):
            q = p if fam == "all" else p[p["family"] == fam]
            lo, lb = q.dropna(subset=["ogt", "pred_ogt"]), q.dropna(subset=["breadth", "pred_breadth"])
            rows.append(dict(model=model, split=split, family=fam, n_ogt=len(lo),
                             ogt_pearson=safe_corr(lo["ogt"], lo["pred_ogt"]), ogt_spearman=safe_corr(lo["ogt"], lo["pred_ogt"], "spearman"),
                             ogt_mae=float(np.abs(lo["ogt"] - lo["pred_ogt"]).mean()) if len(lo) else np.nan,
                             n_spec=len(lb), breadth_pearson=safe_corr(lb["breadth"], lb["pred_breadth"]),
                             breadth_spearman=safe_corr(lb["breadth"], lb["pred_breadth"], "spearman")))
    return pd.DataFrame(rows)


def leakage(runs, ds):
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupKFold, cross_val_predict

    rows = []
    for (model, split), (d, _) in runs.items():
        for f in sorted(d.glob("fold*")):
            if not (f / "zT.npy").exists():
                continue
            zT, zS = np.load(f / "zT.npy"), np.load(f / "zS.npy")
            acc = pd.read_csv(f / "preds.tsv", sep="\t")["accession"]
            meta = ds.set_index("accession").loc[acc]
            rec = dict(model=model, split=split, fold=f.name, dcor_zT_zS=dcor2(zT, zS) if len(zT) > 8 else np.nan)
            for tgt, col in [("ogt", "ogt"), ("breadth", "breadth")]:
                m = meta[col].notna().values
                if m.sum() < 30 or meta.loc[m, "cluster_split"].nunique() < 5:
                    continue
                for zn, Z in [("zT", zT), ("zS", zS)]:
                    cv = GroupKFold(5)
                    pr = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 3, 8)), Z[m], meta.loc[m, col].values,
                                           groups=meta.loc[m, "cluster_split"].values, cv=cv)
                    rec[f"R2_{zn}_to_{tgt}"] = r2_score(meta.loc[m, col].values, pr)
            rows.append(rec)
    return pd.DataFrame(rows)


def tradeoff_tables(runs, ds, n_boot):
    rows = []
    lab = ds.dropna(subset=["ogt", "breadth"])
    for fam in ["all"] + sorted(ds["family"].unique()):
        q = lab if fam == "all" else lab[lab["family"] == fam]
        r = fe_slope(q, "ogt", "breadth", "cluster_pairs", n_boot)
        rows.append(dict(source="observed", family=fam, **r))
    for (model, split), (_, p) in runs.items():
        if split != "cv" or p["pred_ogt"].isna().all():
            continue
        for fam in ["all"] + sorted(p["family"].unique()):
            q = p if fam == "all" else p[p["family"] == fam]
            r = fe_slope(q, "pred_ogt", "pred_breadth", "cluster_pairs", n_boot)
            rows.append(dict(source=f"predicted:{model}", family=fam, **r))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def fig_performance(perf, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = perf[(perf["family"] == "all") & (perf["split"] == "cv")].sort_values("ogt_spearman")
    if d.empty:
        return
    fig, axs = plt.subplots(1, 2, figsize=(9, 0.45 * len(d) + 1.6), facecolor=SURFACE, sharey=True)
    for ax, col, title in [(axs[0], "ogt_spearman", "Temperature (OGT)"), (axs[1], "breadth_spearman", "Specificity breadth")]:
        style(ax)
        colors = [BAR if m.startswith(("disent", "plain", "joint", "noadv", "nodcor")) else BAR_REF for m in d["model"]]
        vals = d[col].fillna(0).values
        ax.barh(d["model"], vals, color=colors, height=0.6)
        for y, v in enumerate(d[col].values):
            ax.text(max(v, 0) + 0.01 if np.isfinite(v) else 0.01, y, "n/a" if not np.isfinite(v) else f"{v:.2f}", va="center", fontsize=8.5, color=INK)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        ax.set_xlabel("Spearman on held-out clusters", fontsize=9, color=INK2)
    fig.suptitle("Held-out accuracy (lineage-aware CV). Blue = neural variants, grey = baselines", x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)


def fig_tradeoff(ds, tbl, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lab = ds.dropna(subset=["ogt", "breadth"]).copy()
    lab = lab[lab.groupby("cluster_pairs")["ogt"].transform("size") >= 2]
    if lab.empty:
        return
    lab["dx"] = lab["ogt"] - lab.groupby("cluster_pairs")["ogt"].transform("mean")
    lab["dy"] = lab["breadth"] - lab.groupby("cluster_pairs")["breadth"].transform("mean")
    fig, ax = plt.subplots(figsize=(6, 4.4), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", color=GRID, lw=0.6)
    for fam, c in FAM_COLORS.items():
        q = lab[lab["family"] == fam]
        ax.scatter(q["dx"], q["dy"], s=18, color=c, alpha=0.75, edgecolor=SURFACE, linewidth=0.8, label=fam)
    r = tbl[(tbl["source"] == "observed") & (tbl["family"] == "all")]
    if len(r) and np.isfinite(r["slope"].iloc[0]):
        s = r["slope"].iloc[0] / 10
        xs = np.array([lab["dx"].min(), lab["dx"].max()])
        ax.plot(xs, s * xs, color=INK, lw=1.5)
        ax.text(0.02, 0.97, f"slope {r['slope'].iloc[0]:+.3f} per +10 °C  [{r['lo'].iloc[0]:+.3f}, {r['hi'].iloc[0]:+.3f}]",
                transform=ax.transAxes, va="top", fontsize=9, color=INK)
    ax.axhline(0, color=GRID, lw=0.8)
    ax.set_xlabel("OGT minus clade mean (°C)", fontsize=9, color=INK2)
    ax.set_ylabel("Specificity breadth minus clade mean", fontsize=9, color=INK2)
    ax.legend(frameon=False, fontsize=9, loc="lower left", title="Family", title_fontsize=9)
    ax.set_title("Within-clade association between growth temperature and specificity breadth", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)


def fig_slopes(tbl, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = tbl[tbl["family"] == "all"].dropna(subset=["slope"]).iloc[::-1]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(d) + 1.4), facecolor=SURFACE)
    style(ax)
    y = np.arange(len(d))
    ax.hlines(y, d["lo"], d["hi"], color=INK2, lw=1.5)
    ax.scatter(d["slope"], y, s=36, color=[BAR if s == "observed" else BAR_REF for s in d["source"]], zorder=3, edgecolor=SURFACE, linewidth=1)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(d["source"], fontsize=9, color=INK)
    ax.set_xlabel("Change in specificity breadth per +10 °C within clades (95% cluster-bootstrap CI)", fontsize=9, color=INK2)
    ax.set_title("Trade-off slope: observed labels (blue) vs held-out model predictions (grey)", loc="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_all(cfg["seed"])
    ds = load_dataset(cfg)
    runs = load_runs(cfg, ds)
    if not runs:
        raise SystemExit("no runs found; run scripts 04/05 first")
    out = rpath(cfg, "runs", "analysis", "x").parent
    figs = rpath(cfg, "figures", "x").parent

    perf = performance(runs)
    perf.to_csv(out / "performance.tsv", sep="\t", index=False)
    print(perf[perf["family"] == "all"].round(3).to_string(index=False))

    leak = leakage(runs, ds)
    if len(leak):
        leak.to_csv(out / "leakage_per_fold.tsv", sep="\t", index=False)
        print("\nLeakage (mean over folds):")
        print(leak.groupby(["model", "split"]).mean(numeric_only=True).round(3).to_string())

    tbl = tradeoff_tables(runs, ds, args.n_boot)
    tbl.to_csv(out / "tradeoff_slopes.tsv", sep="\t", index=False)
    print("\nWithin-clade slope of breadth per +10 C (negative = hotter relatives are more specific):")
    print(tbl.round(4).to_string(index=False))

    fig_performance(perf, figs / "performance.png")
    fig_tradeoff(ds, tbl, figs / "tradeoff_within_clade.png")
    fig_slopes(tbl, figs / "tradeoff_slopes.png")
    json.dump({"n_runs": len(runs)}, open(out / "summary.json", "w"))
    print(f"\nTables -> {out}   Figures -> {figs}")


if __name__ == "__main__":
    main()
