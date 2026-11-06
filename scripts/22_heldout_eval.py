#!/usr/bin/env python
"""Independent test-set evaluation of the temperature regressor.

Bioinformatics states that reporting the average error over cross-validation folds is not sufficient
and that performance must be reported on an independent test set, and that papers using leave-one-out
are editorially rejected. This produces the numbers Section 2.8 of the manuscript needs.

  Design      one fold of the homology-grouped split is held out entirely. It is used neither to fit
              the model nor to choose the ridge penalty. The penalty is chosen by GROUPED k-fold
              inside the training partition, so no leave-one-out anywhere in the chain.
  Homology    folds are grouped by 30%-identity MMseqs2 cluster, so no test sequence has a close
              homologue among the training sequences. The script verifies the partition is clean
              rather than trusting that it is.
  Baseline    the same protocol on plain amino-acid composition, so the reported performance can be
              read against something rather than in a vacuum.

  What this does NOT establish: the paper's claims rest on WITHIN-CLADE slopes, not on between-protein
  prediction. A model can predict growth temperature well and still say nothing about where on a fold
  adaptation acts. This evaluation exists to satisfy a reporting requirement and to show the regressor
  is not noise -- it is not evidence for the pocket result, which has its own permutation tests.

  python scripts/22_heldout_eval.py --config config.yaml
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serprot.common import base_parser, load_config, load_dataset, prepare_inputs, rpath, seed_all  # noqa: E402

AA = "ACDEFGHIKLMNPQRSTVWY"


def model_slug(cfg) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9]+", "_", str(cfg["esm_model"]).rstrip("/").split("/")[-1]).strip("_")


def composition(seqs):
    """20 amino-acid frequencies -- the honest floor any embedding must beat."""
    out = np.zeros((len(seqs), len(AA)), dtype=np.float32)
    idx = {a: i for i, a in enumerate(AA)}
    for r, s in enumerate(seqs):
        if not s:
            continue
        for ch in s:
            j = idx.get(ch)
            if j is not None:
                out[r, j] += 1.0
        out[r] /= max(len(s), 1)
    return out


def metrics(y, p):
    from scipy.stats import pearsonr, spearmanr
    err = p - y
    ss_res = float((err ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    # A constant prediction (the mean baseline) has no correlation to report. Test it with ptp, not
    # std: summation rounds the mean by an ULP, so std of a constant array comes back around 1e-14
    # rather than 0, and a > 0 test on it silently lets the degenerate case through.
    varies = float(np.ptp(p)) > 0 and float(np.ptp(y)) > 0
    return {"n": int(len(y)), "r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
            "pearson_r": float(pearsonr(y, p)[0]) if varies else np.nan,
            "spearman_rho": float(spearmanr(y, p)[0]) if varies else np.nan,
            "mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean()))}


def fit_and_score(Xtr, ytr, gtr, Xte, yte, n_inner, seed):
    """Penalty chosen by GROUPED k-fold inside the training partition only."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV, GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([("sc", StandardScaler()), ("ridge", Ridge(random_state=seed))])
    grid = {"ridge__alpha": np.logspace(-1, 4, 12)}
    n_groups = len(np.unique(gtr))
    cv = GroupKFold(n_splits=min(n_inner, n_groups))
    gs = GridSearchCV(pipe, grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
    gs.fit(Xtr, ytr, groups=gtr)
    best = float(gs.best_params_["ridge__alpha"])
    return gs.best_estimator_.predict(Xte), best


def describe(d, label):
    hot = d[d["ogt"] >= 50]
    return {"partition": label, "proteins": len(d), "clusters_30pct": d["cluster_split"].nunique(),
            "clades_50pct": d["cluster_pairs"].nunique() if "cluster_pairs" in d else np.nan,
            "S8": int((d["family"] == "S8").sum()), "S9": int((d["family"] == "S9").sum()),
            "S1": int((d["family"] == "S1").sum()),
            "thermophiles": len(hot), "pct_thermophile": round(100 * len(hot) / max(len(d), 1), 1),
            "ogt_median": float(d["ogt"].median()), "ogt_min": float(d["ogt"].min()),
            "ogt_max": float(d["ogt"].max())}


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--test-fold", type=int, default=0, help="which fold_cv value is held out entirely")
    ap.add_argument("--inner-folds", type=int, default=5, help="grouped folds for penalty selection")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_all(cfg["seed"])

    full = prepare_inputs(load_dataset(cfg), cfg["max_len"])
    cache = Path(rpath(cfg, "processed", f"esm_embeddings_{model_slug(cfg)}.npy"))
    if not cache.exists():
        raise SystemExit(f"need the stage-4 embedding cache at {cache}; run 04_embed_baselines.py first")
    E = np.load(cache)
    if len(E) != len(full):
        raise SystemExit(f"embedding cache has {len(E)} rows against {len(full)} proteins; "
                         "re-run 04_embed_baselines.py so the two line up")
    for col in ["fold_cv", "cluster_split", "ogt", "family"]:
        if col not in full:
            raise SystemExit(f"dataset.tsv has no {col} column; run 03_cluster_split.py first")

    keep = full["ogt"].notna().values & full["fold_cv"].notna().values
    full, E = full[keep].reset_index(drop=True), E[keep]
    te = (full["fold_cv"].astype(int) == args.test_fold).values
    tr = ~te
    print(f"embeddings {E.shape}   training {tr.sum()}   held out {te.sum()}")

    # the partition is only homology-safe if no 30% cluster appears on both sides -- check it
    shared = set(full.loc[tr, "cluster_split"]) & set(full.loc[te, "cluster_split"])
    print(f"30%-identity clusters shared between training and test: {len(shared)}")
    if shared:
        raise SystemExit("the held-out fold shares clusters with training; the split is not "
                         "homology-safe and the evaluation would be invalid")

    comp = pd.DataFrame([describe(full[tr], "training"), describe(full[te], "held-out test")])
    print("\n===== dataset composition =====")
    print(comp.to_string(index=False))

    y = full["ogt"].values.astype(float)
    g = full["cluster_split"].values
    rows = []
    print("\n===== held-out performance =====")
    print(f"{'features':>16}{'n':>7}{'R2':>8}{'Pearson':>9}{'Spearman':>10}{'MAE':>8}{'RMSE':>8}{'alpha':>9}")
    preds = {}
    for name, X in [("ESM-2 650M", E), ("AA composition", composition(full["seq_in"].tolist()))]:
        p, alpha = fit_and_score(X[tr], y[tr], g[tr], X[te], y[te], args.inner_folds, cfg["seed"])
        m = metrics(y[te], p)
        preds[name] = p
        rows.append({"features": name, "partition": "held-out test", **m, "alpha": alpha})
        print(f"{name:>16}{m['n']:>7}{m['r2']:>8.3f}{m['pearson_r']:>9.3f}"
              f"{m['spearman_rho']:>10.3f}{m['mae']:>8.2f}{m['rmse']:>8.2f}{alpha:>9.3g}")
    base = metrics(y[te], np.full(te.sum(), y[tr].mean()))
    rows.append({"features": "training mean", "partition": "held-out test", **base, "alpha": np.nan})
    print(f"{'training mean':>16}{base['n']:>7}{base['r2']:>8.3f}{'':>9}{'':>10}"
          f"{base['mae']:>8.2f}{base['rmse']:>8.2f}")

    print("\n===== held-out performance by family (ESM-2) =====")
    print(f"{'family':>8}{'n':>7}{'R2':>8}{'Spearman':>10}{'MAE':>8}")
    for fam in ["S8", "S9", "S1"]:
        sel = (full.loc[te, "family"] == fam).values
        if sel.sum() < 30:
            print(f"{fam:>8}{int(sel.sum()):>7}   too few held-out proteins to score")
            continue
        m = metrics(y[te][sel], preds["ESM-2 650M"][sel])
        rows.append({"features": "ESM-2 650M", "partition": f"held-out test / {fam}", **m,
                     "alpha": np.nan})
        print(f"{fam:>8}{m['n']:>7}{m['r2']:>8.3f}{m['spearman_rho']:>10.3f}{m['mae']:>8.2f}")

    out = rpath(cfg, "runs", "analysis", "heldout_eval.tsv")
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
    cout = rpath(cfg, "runs", "analysis", "heldout_composition.tsv")
    comp.to_csv(cout, sep="\t", index=False)
    print(f"\nwrote {out}\nwrote {cout}")
    print("\nFor Section 2.8, quote the ESM-2 row: R2, Spearman and MAE on the held-out partition,")
    print("against the training-mean baseline. Say plainly that this is a between-protein prediction")
    print("metric and that the paper's claims rest on within-clade slopes, which it does not test.")


if __name__ == "__main__":
    main()
