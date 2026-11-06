#!/usr/bin/env python
"""Is the pocket exclusion a general property of prokaryotic serine proteases, or is it carried by one
taxonomic group?

Archaea and bacteria reached high growth temperatures by partly different routes, and thermophilic
archaea are over-represented among the hot species in any OGT table. If the whole pocket-vs-control
result were an archaeal result wearing a prokaryotic label, that would matter, and the paper should say
so. This splits the SAME descriptor table stage 12 analysed, runs the SAME statistics inside each
subgroup, and then tests whether the pocket-control difference itself differs between subgroups.

No new structures, no new embeddings -- this reads what stages 10-11 already wrote.

  Read the result:
    the difference has the same sign and overlapping intervals in both groups, interaction p large
      -> one prokaryotic result, reported as such
    the difference is present in one group and absent in the other, interaction p small
      -> a group-specific result, and the claim must be narrowed to that group

  python scripts/19_subgroups.py --config config.yaml
  python scripts/19_subgroups.py --config config.yaml --tag _esmfold
  python scripts/19_subgroups.py --config config.yaml --split phylum --top 4
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serprot.common import base_parser, load_config, load_dataset, rpath, seed_all  # noqa: E402
from serprot.stats import clade_bootstrap, permute_within_clades, slope  # noqa: E402

DESCRIPTORS = ["resvol", "f_polar", "f_pro", "f_ivywrel", "f_neg", "hydropathy"]


def domain_of(lineage: str) -> str:
    """'cellular organisms (no rank), Archaea (superkingdom), ...' -> 'Archaea'."""
    if not isinstance(lineage, str):
        return "Unknown"
    m = re.search(r"\b(Archaea|Bacteria|Eukaryota)\b", lineage)
    return m.group(1) if m else "Unknown"


def load_descriptors(cfg, tag: str, radius: int, min_plddt: float, allow_partial: bool) -> pd.DataFrame:
    """The same files, filters and shard guard stage 12 uses, so the subgroup rows are its rows."""
    rx = re.compile(rf"^pocket_descriptors{re.escape(tag)}(\.\d+)?\.tsv$")
    allf = sorted(Path(rpath(cfg, "processed", "x")).parent.glob("pocket_descriptors*.tsv"))
    parts = [p for p in allf if rx.match(p.name)]
    print(f"descriptor files for tag {tag!r}: {[p.name for p in parts]}")
    if not parts:
        raise SystemExit("no matching pocket_descriptors*.tsv; run 11_pocket_descriptors.py first")
    frames, declared = [], []
    for f in parts:
        fr = pd.read_csv(f, sep="\t")
        if "n_shards" in fr and len(fr) and pd.notna(fr["n_shards"].iloc[0]):
            declared.append(int(fr["n_shards"].iloc[0]))
        frames.append(fr)
    d = pd.concat(frames, ignore_index=True).drop_duplicates("accession")
    shards = sorted(int(f.stem.split(".")[-1]) for f in parts if f.stem.split(".")[-1].isdigit())
    expected = max(declared) if declared else (max(shards) + 1 if shards else 1)
    missing = [i for i in range(expected) if i not in shards] if shards else []
    if missing and not allow_partial:
        raise SystemExit(f"incomplete descriptor set: have {shards}, missing {missing}. "
                         "Pass --allow-partial only if you mean to analyse a subset.")
    tg = pd.read_csv(rpath(cfg, "processed", "structure_targets.tsv"), sep="\t")
    df = d.merge(tg[["accession", "family", "cluster_pairs", "ogt"]], on="accession", how="inner")
    if "status" in df:
        df = df[df["status"] == "ok"]
    pk_plddt = f"pk{radius}_plddt"
    if pk_plddt in df:
        df = df[df[pk_plddt] >= min_plddt]
    good = df.groupby("cluster_pairs")["ogt"].transform("size") >= 2
    spread = df.groupby("cluster_pairs")["ogt"].transform(lambda s: s.max() - s.min()) > 0
    return df[good & spread].reset_index(drop=True)


def diff_for(mask, yp, yc, x, codes, k):
    m = np.flatnonzero(mask)
    if len(m) < 10:
        return np.nan
    return slope(yp[m], x[m], codes[m], k) - slope(yc[m], x[m], codes[m], k)


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--radius", type=int, default=10)
    ap.add_argument("--min-plddt", type=float, default=70.0)
    ap.add_argument("--split", choices=["domain", "phylum"], default="domain")
    ap.add_argument("--top", type=int, default=2, help="how many groups to keep when splitting on phylum")
    ap.add_argument("--min-clades", type=int, default=25, help="skip a group with fewer clades than this")
    ap.add_argument("--descriptors", default=",".join(DESCRIPTORS))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--tag", default="", help="suffix identifying the structure set, e.g. _esmfold")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_all(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    df = load_descriptors(cfg, args.tag, args.radius, args.min_plddt, args.allow_partial)

    ds = load_dataset(cfg)
    need = ["accession", "lineage"] + (["phylum"] if args.split == "phylum" else [])
    missing_cols = [c for c in need if c not in ds]
    if missing_cols:
        raise SystemExit(f"dataset.tsv has no {missing_cols}; it is written by 01_fetch_sequences.py, "
                         "which stores the UniProt lineage string the split is derived from")
    meta = ds[need]
    df = df.merge(meta, on="accession", how="left")
    df["group"] = df["phylum"].fillna("Unknown") if args.split == "phylum" else df["lineage"].map(domain_of)

    counts = (df.groupby("group")
                .agg(proteins=("accession", "size"), clades=("cluster_pairs", "nunique"),
                     thermophiles=("ogt", lambda s: int((s >= 50).sum())),
                     median_ogt=("ogt", "median"))
                .sort_values("clades", ascending=False))
    print(f"\n===== split on {args.split} =====")
    print(counts.to_string())

    keep = [g for g in counts.index if g != "Unknown" and counts.loc[g, "clades"] >= args.min_clades]
    keep = keep[: args.top] if args.split == "phylum" else keep
    if len(keep) < 2:
        raise SystemExit(f"need two groups with >= {args.min_clades} clades; got {keep}")
    print(f"\ncomparing: {keep}")
    for g in keep:
        sub = df[df["group"] == g]
        if sub["ogt"].max() - sub["ogt"].min() < 20:
            print(f"WARNING: {g} spans only {sub['ogt'].min():.0f}-{sub['ogt'].max():.0f} C; "
                  "a slope over that range is weakly identified")

    # one clade label per clade: a 50%-identity cluster is effectively within one group, but take the
    # majority rather than assuming it, and report any clade that is actually mixed
    clades, codes = np.unique(df["cluster_pairs"].values, return_inverse=True)
    k = len(clades)
    by_clade = df.assign(_c=codes).groupby("_c")["group"]
    clade_grp = by_clade.agg(lambda s: s.mode().iloc[0]).reindex(range(k)).values.astype(object)
    mixed = int((by_clade.nunique().reindex(range(k)).fillna(1) > 1).sum())
    print(f"{mixed} of {k} clades contain more than one {args.split}; majority label used for those")

    x = df["ogt"].values.astype(float)
    rows = []
    for base in args.descriptors.split(","):
        pk, ct = f"pk{args.radius}_{base}", f"ct_{base}"
        if pk not in df or ct not in df:
            print(f"\n-- {base}: columns missing, skipped")
            continue
        yp = df[pk].values.astype(float)
        yc = df[ct].values.astype(float)
        print(f"\n===== {base}: within-clade slope on OGT per +10 C, by {args.split} =====")
        print(f"{'group':>22}{'pocket':>13}{'control':>13}{'difference':>14}"
              f"{'95% CI':>26}{'p':>9}{'n':>8}{'clades':>8}")
        per_group = {}
        for g in keep:
            m = np.flatnonzero(df["group"].values == g)
            cg = np.unique(codes[m], return_inverse=True)[1]   # clade codes renumbered inside the group
            kg = int(cg.max()) + 1
            ypg, ycg, xg = yp[m], yc[m], x[m]
            sp = slope(ypg, xg, cg, kg)
            sc = slope(ycg, xg, cg, kg)
            dd = sp - sc
            bs = clade_bootstrap(
                lambda i2, rc: slope(ypg[i2], xg[i2], rc, kg) - slope(ycg[i2], xg[i2], rc, kg),
                cg, kg, args.n_boot, rng)
            lo, hi = (np.percentile(bs, [2.5, 97.5]) if len(bs) > 10 else (np.nan, np.nan))
            null = []
            for _ in range(args.n_perm):
                xs = permute_within_clades(xg, cg, kg, rng)
                null.append(slope(ypg, xs, cg, kg) - slope(ycg, xs, cg, kg))
            null = np.array([v for v in null if np.isfinite(v)])
            p = float((np.abs(null) >= abs(dd)).mean()) if len(null) else np.nan
            ci = f"[{lo:+.5f}, {hi:+.5f}]"
            print(f"{g:>22}{sp:>+13.5f}{sc:>+13.5f}{dd:>+14.5f}{ci:>26}{p:>9.4f}{len(m):>8}{kg:>8}")
            per_group[g] = dd
            rows.append({"descriptor": base, "split": args.split, "group": g, "slope_pocket": sp,
                         "slope_control": sc, "difference": dd, "difference_lo": lo,
                         "difference_hi": hi, "p_perm": p, "n": len(m), "clades": kg})

        # interaction: is the pocket-control difference itself different between the two groups?
        # Clade labels are shuffled between groups, so clade structure and every within-clade slope
        # survive; only which group a clade belongs to is randomised.
        ga, gb = keep[0], keep[1]
        obs = per_group.get(ga, np.nan) - per_group.get(gb, np.nan)
        if np.isfinite(obs):
            null = []
            for _ in range(args.n_perm):
                cg = rng.permutation(clade_grp)
                row_grp = cg[codes]
                null.append(diff_for(row_grp == ga, yp, yc, x, codes, k) -
                            diff_for(row_grp == gb, yp, yc, x, codes, k))
            null = np.array([v for v in null if np.isfinite(v)])
            pi = float((np.abs(null) >= abs(obs)).mean()) if len(null) else np.nan
            pair = f"{ga} - {gb}"
            print(f"{'interaction':>22}  ({pair}) difference-of-differences {obs:+.5f}   p={pi:.4f}")
            rows.append({"descriptor": base, "split": args.split, "group": f"{ga}_vs_{gb}",
                         "difference": obs, "p_perm": pi, "n": len(df), "clades": k})

    out = rpath(cfg, "runs", "analysis", f"subgroups_{args.split}{args.tag}.tsv")
    pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
    print(f"\nwrote {out}")
    print("\nA large interaction p does not prove the groups are the same; with this many clades it")
    print("means the data cannot distinguish them, which is the honest thing to report.")


if __name__ == "__main__":
    main()
