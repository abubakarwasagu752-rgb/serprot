#!/usr/bin/env python
"""Choose a small panel of natural homologs for wet-lab validation (Aim 4).

Design: within one clade of relatives (cluster_pairs), a 2x2 of {hot, cool} x {narrow, broad} predicted specificity.
Clades where all four cells are populated and the predicted spreads are large give the cleanest test of the trade-off:
if hot relatives are systematically narrower, the (hot, broad) and (cool, narrow) cells are the informative minority.
Uses held-out predictions only. Output: runs/analysis/candidates.tsv  (measure Tm/Topt and cleavage profile for each).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serprot.common import base_parser, load_config, load_dataset, rpath  # noqa: E402


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--model", default="disent")
    ap.add_argument("--split", default="cv")
    ap.add_argument("--n-clades", type=int, default=12)
    ap.add_argument("--min-ogt-spread", type=float, default=12.0)
    ap.add_argument("--min-breadth-spread", type=float, default=0.04)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ds = load_dataset(cfg)
    run = rpath(cfg, "runs", f"{args.model}_{args.split}")
    p = pd.concat([pd.read_csv(f / "preds.tsv", sep="\t") for f in sorted(run.glob("fold*"))])
    d = p.merge(ds, on="accession")
    d = d[d["nuc_pos"].notna()]

    rows = []
    for cl, g in d.groupby("cluster_pairs"):
        if len(g) < 4:
            continue
        s_ogt = g["pred_ogt"].max() - g["pred_ogt"].min()
        s_br = g["pred_breadth"].max() - g["pred_breadth"].min()
        if s_ogt < args.min_ogt_spread or s_br < args.min_breadth_spread:
            continue
        hot = g["pred_ogt"] >= g["pred_ogt"].median()
        broad = g["pred_breadth"] >= g["pred_breadth"].median()
        cells = {"hot_broad": g[hot & broad], "hot_narrow": g[hot & ~broad], "cool_broad": g[~hot & broad], "cool_narrow": g[~hot & ~broad]}
        if any(len(v) == 0 for v in cells.values()):
            continue
        score = min(s_ogt / 20, 1.0) * min(s_br / 0.10, 1.0)
        # informativeness: reward the cells that CONTRADICT the usual trade-off direction having members
        for name, c in cells.items():
            pick = c.sort_values("pred_ogt", ascending=name.startswith("cool")).iloc[0]
            rows.append(dict(clade=cl, family=pick["family"], clade_score=score, cell=name, accession=pick["accession"],
                             organism=pick["organism"], pred_ogt=pick["pred_ogt"], pred_breadth=pick["pred_breadth"],
                             obs_ogt=pick["ogt"], obs_breadth=pick["breadth"], length=pick["length"],
                             nuc_source=pick["nuc_source"], clade_size=len(g), ogt_spread=s_ogt, breadth_spread=s_br))
    if not rows:
        raise SystemExit("no clade satisfies the spread thresholds; relax --min-ogt-spread / --min-breadth-spread")
    out = pd.DataFrame(rows)
    top = out.drop_duplicates("clade").sort_values("clade_score", ascending=False).head(args.n_clades)["clade"]
    out = out[out["clade"].isin(top)].sort_values(["clade_score", "clade", "cell"], ascending=[False, True, True])
    path = rpath(cfg, "runs", "analysis", "candidates.tsv")
    out.to_csv(path, sep="\t", index=False)
    print(out.groupby("family")["clade"].nunique().rename("clades").to_string())
    print(f"{len(out)} proteins in {out['clade'].nunique()} clades -> {path}")


if __name__ == "__main__":
    main()
