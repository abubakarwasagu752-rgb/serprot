#!/usr/bin/env python
"""Pick the proteins worth downloading AlphaFold models for.

The pocket analysis compares relatives inside a 50%-identity clade that differ in growth temperature,
so a protein is only useful if (a) its catalytic Ser is known (the pocket is defined from it), (b) it has
an OGT label, and (c) it sits in a clade where other members have OGT labels spanning a real range.

Run with no thresholds to print a sensitivity table first, then re-run with the chosen values.

  python scripts/09_select_structures.py --config config.yaml                    # table only
  python scripts/09_select_structures.py --config config.yaml --min-members 3 --min-spread 10 --write
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serprot.common import base_parser, load_config, load_dataset, rpath, seed_all  # noqa: E402


def clade_stats(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("cluster_pairs")["ogt"]
    s = pd.DataFrame({"n_ogt": g.size(), "spread": g.max() - g.min(), "ogt_min": g.min(), "ogt_max": g.max()})
    fam = df.groupby("cluster_pairs")["family"].agg(lambda x: x.mode().iat[0])
    s["family"] = fam
    s["n_hot"] = df.assign(h=df["ogt"] >= 50).groupby("cluster_pairs")["h"].sum()
    return s


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--min-members", type=int, default=None, help="OGT-labeled members a clade must have")
    ap.add_argument("--min-spread", type=float, default=None, help="min (max-min) OGT within the clade, in C")
    ap.add_argument("--max-per-clade", type=int, default=40, help="cap; members are chosen to span the OGT range")
    ap.add_argument("--write", action="store_true", help="write the accession list (otherwise table only)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_all(cfg["seed"])

    df = load_dataset(cfg)
    n0 = len(df)
    df = df[df["nuc_pos"].notna() & df["ogt"].notna()].copy()
    df["ogt"] = df["ogt"].astype(float)
    print(f"{n0} sequences -> {len(df)} with BOTH a catalytic Ser and an OGT label")
    print(df.groupby("family").agg(n=("accession", "size"), hot=("ogt", lambda s: int((s >= 50).sum()))).to_string())

    st = clade_stats(df)
    print(f"\nclades (50% identity) containing such proteins: {len(st)}")

    spreads = (0, 5, 10, 15, 20)
    print("\nsensitivity table -- each cell: clades / proteins to download / of those thermophilic (>=50 C)")
    print(f"{'min OGT spread within clade:':<30}" + "".join(f"{str(sp) + ' C':>22}" for sp in spreads))
    for m in (2, 3, 4, 5, 8):
        cells = []
        for sp in spreads:
            keep = st[(st["n_ogt"] >= m) & (st["spread"] >= sp)]
            sub = df[df["cluster_pairs"].isin(keep.index)]
            cells.append(f"{len(keep)} / {len(sub)} / {int((sub['ogt'] >= 50).sum())}")
        print(f"{'  >= ' + str(m) + ' labeled members':<30}" + "".join(f"{c:>22}" for c in cells))

    if args.min_members is None or args.min_spread is None:
        print("\nNo thresholds given. Re-run with --min-members and --min-spread (and --write) once chosen.")
        return

    keep = st[(st["n_ogt"] >= args.min_members) & (st["spread"] >= args.min_spread)]
    sub = df[df["cluster_pairs"].isin(keep.index)].copy()
    # cap large clades: keep the extremes of the OGT range, then spread the rest evenly
    rng = np.random.default_rng(cfg["seed"])
    parts = []
    for cl, g in sub.groupby("cluster_pairs"):
        if len(g) <= args.max_per_clade:
            parts.append(g)
            continue
        g = g.sort_values("ogt")
        idx = np.unique(np.linspace(0, len(g) - 1, args.max_per_clade).round().astype(int))
        parts.append(g.iloc[idx])
    sel = pd.concat(parts).sort_values(["family", "cluster_pairs", "ogt"])

    print(f"\nselected: {len(keep)} clades, {len(sel)} proteins "
          f"({int((sel['ogt'] >= 50).sum())} thermophilic), capped at {args.max_per_clade}/clade")
    print(sel.groupby("family").agg(n=("accession", "size"), clades=("cluster_pairs", "nunique"),
                                    hot=("ogt", lambda s: int((s >= 50).sum()))).to_string())
    print(f"\nestimated AlphaFold download: ~{len(sel) * 0.35:.0f} MB at ~350 KB/model")

    if args.write:
        out = rpath(cfg, "processed", "structure_targets.tsv")
        sel[["accession", "family", "cluster_pairs", "ogt", "nuc_pos", "length", "organism"]].to_csv(out, sep="\t", index=False)
        lst = rpath(cfg, "processed", "structure_accessions.txt")
        Path(lst).write_text("\n".join(sel["accession"]) + "\n")
        print(f"\nwrote {out}\nwrote {lst}")


if __name__ == "__main__":
    main()
