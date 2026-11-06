#!/usr/bin/env python
"""Does thermal adaptation change the substrate-binding pocket MORE, or differently, than the rest of the protein?

For each descriptor the within-clade slope on OGT is computed with clade (50% identity cluster) fixed effects,
separately for the pocket and for the matched control patches. The quantity of interest is the DIFFERENCE,
because a protein-wide thermal trend moves both.

Uncertainty comes from two places, since the control patches are matched but not perfect:
  bootstrap    clades resampled with replacement -> CI on each slope and on the difference
  permutation  OGT shuffled WITHIN clades -> null distribution of the difference (assumption-free p-value)
Benjamini-Hochberg FDR is applied across descriptors.

  python scripts/12_pocket_analysis.py --config config.yaml --n-boot 2000 --n-perm 2000
"""
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serprot.common import base_parser, load_config, load_dataset, rpath, seed_all  # noqa: E402

PAIRED = ["n", "hydropathy", "resvol", "f_pos", "f_neg", "net_charge", "f_aromatic", "f_polar", "f_small",
          "f_gly", "f_pro", "f_ivywrel", "sasa_mean", "sasa_total", "burial", "rgyr", "plddt"]
POCKET_ONLY = ["cavity_open_vol", "cavity_enclosed_vol"]


def demean(v: np.ndarray, codes: np.ndarray, k: int) -> np.ndarray:
    """subtract the clade mean from v (clade fixed effects)"""
    sums = np.bincount(codes, weights=v, minlength=k)
    cnts = np.bincount(codes, minlength=k).astype(float)
    return v - (sums / np.maximum(cnts, 1))[codes]


def within_sd(y: np.ndarray, codes: np.ndarray, k: int) -> float:
    """SD of y after removing clade means -- the variability actually available to the regression."""
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    return float(demean(y[ok], codes[ok], k).std())


def slope(y: np.ndarray, x: np.ndarray, codes: np.ndarray, k: int, z: np.ndarray | None = None) -> float:
    """Within-clade slope of y on x, per +10 C. If z is given (e.g. pLDDT), it is partialled out first:
    AlphaFold accuracy tracks alignment depth, which tracks how well-studied an organism is, which tracks
    temperature -- so model quality is a candidate confound and must be removable."""
    ok = np.isfinite(y) & np.isfinite(x)
    if z is not None:
        ok &= np.isfinite(z)
    if ok.sum() < 5:
        return np.nan
    yc, xc = demean(y[ok], codes[ok], k), demean(x[ok], codes[ok], k)
    if z is not None:
        zc = demean(z[ok], codes[ok], k)
        zz = (zc * zc).sum()
        if zz > 0:                      # residualise both on the confound, then regress
            yc = yc - zc * (zc * yc).sum() / zz
            xc = xc - zc * (zc * xc).sum() / zz
    den = (xc * xc).sum()
    return float((xc * yc).sum() / den * 10) if den > 0 else np.nan


def analyse(df: pd.DataFrame, pk: str, ct: str | None, n_boot: int, n_perm: int, rng,
            zcol: str | None = None, standardize: bool = False) -> dict:
    clades, codes = np.unique(df["cluster_pairs"].values, return_inverse=True)
    k = len(clades)
    x = df["ogt"].values.astype(float)
    yp = df[pk].values.astype(float)
    yc = df[ct].values.astype(float) if ct else None
    z = df[zcol].values.astype(float) if zcol and zcol in df else None
    sd_p = within_sd(yp, codes, k)
    sd_c = within_sd(yc, codes, k) if ct else np.nan
    if standardize:
        # express each descriptor in units of its OWN within-clade SD, so a pocket that simply varies
        # less than a surface patch is not mistaken for a pocket that resists temperature
        if np.isfinite(sd_p) and sd_p > 0:
            yp = yp / sd_p
        if ct and np.isfinite(sd_c) and sd_c > 0:
            yc = yc / sd_c
    obs_p = slope(yp, x, codes, k, z)
    obs_c = slope(yc, x, codes, k, z) if ct else np.nan
    obs_d = obs_p - obs_c if ct else np.nan

    # cluster bootstrap: resample whole clades
    members = [np.flatnonzero(codes == i) for i in range(k)]
    bp, bc, bd = [], [], []
    for _ in range(n_boot):
        pick = rng.integers(0, k, k)
        idx = np.concatenate([members[i] for i in pick])
        rc = np.repeat(np.arange(k), [len(members[i]) for i in pick])
        zi = z[idx] if z is not None else None
        sp = slope(yp[idx], x[idx], rc, k, zi)
        bp.append(sp)
        if ct:
            sc = slope(yc[idx], x[idx], rc, k, zi)
            bc.append(sc)
            bd.append(sp - sc)

    def ci(v):
        v = np.array([z for z in v if np.isfinite(z)])
        return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) > 10 else (np.nan, np.nan)

    out = {"descriptor": pk.split("_", 1)[-1] if pk.startswith("pk") else pk,
           "n": int(np.isfinite(yp).sum()), "clades": k,
           "within_sd_pocket": sd_p, "within_sd_control": sd_c,
           "slope_pocket": obs_p, "slope_pocket_lo": ci(bp)[0], "slope_pocket_hi": ci(bp)[1]}
    if ct:
        out.update({"slope_control": obs_c, "slope_control_lo": ci(bc)[0], "slope_control_hi": ci(bc)[1],
                    "difference": obs_d, "difference_lo": ci(bd)[0], "difference_hi": ci(bd)[1]})

    # permutation: shuffle OGT within clades, so clade structure and marginal OGT distribution are preserved
    stat = obs_d if ct else obs_p
    if n_perm and np.isfinite(stat):
        null = []
        for _ in range(n_perm):
            xs = x.copy()
            for mem in members:
                if len(mem) > 1:
                    xs[mem] = rng.permutation(xs[mem])
            s = slope(yp, xs, codes, k, z)
            if ct:
                s -= slope(yc, xs, codes, k, z)
            null.append(s)
        null = np.array([z for z in null if np.isfinite(z)])
        out["p_perm"] = float((np.abs(null) >= abs(stat)).mean()) if len(null) else np.nan
        out["null_sd"] = float(null.std()) if len(null) else np.nan
    return out


_SHARED: dict = {}


def _job(args):
    """One (family, descriptor, adjustment) cell. Each gets its own seeded RNG so results are
    reproducible and independent of how the work is scheduled."""
    fam, adj_name, zcol, std, pk, ct, n_boot, n_perm, seed = args
    df = _SHARED["df"]
    sub = df if fam == "all" else df[df["family"] == fam]
    rng = np.random.default_rng(seed)
    return {"family": fam, "adjustment": adj_name,
            **analyse(sub, pk, ct, n_boot, n_perm, rng, zcol, std)}


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    ok = np.isfinite(p)
    q = np.full(len(p), np.nan)
    v = p[ok]
    order = np.argsort(v)
    ranked = v[order] * len(v) / (np.arange(len(v)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(len(v))
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--radius", type=int, default=10, help="which pocket shell to analyse (6, 8 or 10)")
    ap.add_argument("--min-plddt", type=float, default=70.0, help="drop proteins whose pocket is low-confidence")
    ap.add_argument("--min-clade-members", type=int, default=2)
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel analyses; 0 = use SLURM_CPUS_PER_TASK (or all cores)")
    ap.add_argument("--tag", default="", help='suffix added to output filenames (e.g. _esmfold) so runs do not overwrite each other')
    ap.add_argument("--allow-partial", action="store_true",
                    help="analyse even when the descriptor shard files look incomplete")
    ap.add_argument("--no-standardized", action="store_true",
                    help="skip the pass that divides each descriptor by its own within-clade SD")
    ap.add_argument("--no-plddt-control", action="store_true",
                    help="skip the pLDDT-adjusted pass (it doubles runtime)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_all(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    # match exactly "pocket_descriptors<tag>.tsv" or "pocket_descriptors<tag>.<shard>.tsv", so an
    # untagged run never picks up a tagged one (e.g. the ESMFold replication) or vice versa
    rx = re.compile(rf"^pocket_descriptors{re.escape(args.tag)}(\.\d+)?\.tsv$")
    allf = sorted(Path(rpath(cfg, "processed", "x")).parent.glob("pocket_descriptors*.tsv"))
    parts = [p for p in allf if rx.match(p.name)]
    print(f"descriptor files for tag {args.tag!r}: {[p.name for p in parts]}"
          f"   (ignored: {[p.name for p in allf if p not in parts]})")
    if not parts:
        raise SystemExit("no pocket_descriptors*.tsv found; run 11_pocket_descriptors.py first")
    frames, declared_shards = [], []
    for f in parts:
        fr = pd.read_csv(f, sep="\t")
        if "n_shards" in fr and len(fr) and pd.notna(fr["n_shards"].iloc[0]):
            declared_shards.append(int(fr["n_shards"].iloc[0]))   # read per FILE, before de-duplication
        frames.append(fr)
    d = pd.concat(frames, ignore_index=True).drop_duplicates("accession")
    shards = sorted(int(f.stem.split(".")[-1]) for f in parts if f.stem.split(".")[-1].isdigit())
    # tolerate a mix of files written before and after n_shards was recorded: take the largest
    # value any file reports, and fall back to the highest shard index seen
    expected = max(declared_shards) if declared_shards else (max(shards) + 1 if shards else 1)
    missing = [i for i in range(expected) if i not in shards] if shards else []
    if shards:
        print(f"descriptor shards present: {shards} of {expected} expected")
    if (missing or (shards and shards != list(range(len(shards))))) and not args.allow_partial:
        raise SystemExit(
            f"incomplete descriptor set: have shards {shards}, expected 0..{expected - 1}, missing {missing}.\n"
            "A shard job is probably still running or failed. Analysing part of the data would silently "
            "change every result, so this is refused. Wait for the array to finish, or pass --allow-partial "
            "if you really mean to analyse a subset.")
    tg = pd.read_csv(rpath(cfg, "processed", "structure_targets.tsv"), sep="\t")
    df = d.merge(tg[["accession", "family", "cluster_pairs", "ogt"]], on="accession", how="inner")
    print(f"{len(d)} descriptor rows from {len(parts)} file(s); {len(df)} joined to targets")

    if "status" in df:
        print(df["status"].value_counts().to_string())
        df = df[df["status"] == "ok"]
    pk_prefix = f"pk{args.radius}_"
    qc = df[f"{pk_prefix}plddt"] >= args.min_plddt
    print(f"dropped {int((~qc).sum())} proteins with pocket pLDDT < {args.min_plddt}")
    df = df[qc]
    good = df.groupby("cluster_pairs")["ogt"].transform("size") >= args.min_clade_members
    spread = df.groupby("cluster_pairs")["ogt"].transform(lambda s: s.max() - s.min()) > 0
    df = df[good & spread]
    print(f"analysing {len(df)} proteins in {df['cluster_pairs'].nunique()} clades "
          f"({int((df['ogt'] >= 50).sum())} thermophilic)\n")

    adjustments = [("raw", None, False)]
    if not args.no_plddt_control:
        adjustments.append(("plddt_adj", f"{pk_prefix}plddt", False))
    if not args.no_standardized:
        adjustments.append(("standardized", None, True))

    jobs = []
    for adj_name, zcol, std in adjustments:
        for fam in ["all"] + sorted(df["family"].unique()):
            sub = df if fam == "all" else df[df["family"] == fam]
            if sub["cluster_pairs"].nunique() < 5:
                continue
            for base in PAIRED:
                pk, ct = f"{pk_prefix}{base}", f"ct_{base}"
                if pk in sub and ct in sub and not (zcol and base == "plddt"):
                    jobs.append((fam, adj_name, zcol, std, pk, ct))
            for base in POCKET_ONLY:
                if base in sub:
                    jobs.append((fam, adj_name, zcol, std, base, None))
    jobs = [(*j, args.n_boot, args.n_perm, cfg["seed"] + 1000 * i) for i, j in enumerate(jobs)]

    _SHARED["df"] = df
    nw = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    nw = max(1, min(nw, len(jobs)))
    print(f"{len(jobs)} analyses ({args.n_boot} bootstrap + {args.n_perm} permutation each) on {nw} workers",
          flush=True)
    t_an = time.time()
    if nw == 1:
        rows = [_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=nw) as ex:
            rows = list(ex.map(_job, jobs, chunksize=1))
    print(f"done in {(time.time() - t_an) / 60:.1f} min")
    res = pd.DataFrame(rows)
    res.insert(0, "radius", args.radius)
    res["q_fdr"] = np.nan
    for (fam, _adj), g in res.groupby(["family", "adjustment"]):
        res.loc[g.index, "q_fdr"] = bh(g["p_perm"].values)
    out = rpath(cfg, "runs", "analysis", f"pocket_vs_temperature{args.tag}_r{args.radius}.tsv")
    res.to_csv(out, sep="\t", index=False)

    show = ["descriptor", "slope_pocket", "slope_control", "difference", "difference_lo", "difference_hi",
            "p_perm", "q_fdr", "n", "clades"]
    for (fam, adj), g in res.groupby(["family", "adjustment"]):
        g = g.sort_values("p_perm")
        note = {"raw": "",
                "plddt_adj": "   [pLDDT partialled out: guards against model-quality confounding]",
                "standardized": "   [each descriptor in units of its own within-clade SD: guards against "
                                "the pocket simply varying less than a surface patch]"}.get(adj, "")
        print(f"===== {fam} / {adj} / pocket radius {args.radius} A :  "
              f"within-clade change per +10 C  (difference = pocket - control){note} =====")
        print(g[[c for c in show if c in g]].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        print()
    print(f"wrote {out}")

    # ---- validation anchor: do pocket descriptors track the few MEASURED specificities we have?
    try:
        ds = load_dataset(cfg)[["accession", "breadth", "spec_n"]]
        v = df.merge(ds.dropna(subset=["breadth"]), on="accession")
        if len(v) >= 5:
            print(f"\nMEROPS anchor: {len(v)} proteins with measured cleavage breadth")
            cc = [(b, float(v[f"{pk_prefix}{b}"].corr(v["breadth"], method="spearman")))
                  for b in ["resvol", "hydropathy", "f_aromatic", "n", "sasa_total"] if f"{pk_prefix}{b}" in v]
            for b, r in sorted(cc, key=lambda t: -abs(t[1])):
                print(f"   spearman(pocket {b}, measured breadth) = {r:+.2f}")
            print("   (n is small; treat as a sanity check, not evidence)")
        else:
            print(f"\nMEROPS anchor: only {len(v)} proteins have both a structure and measured breadth; skipped")
    except Exception as e:  # noqa: BLE001
        print(f"\nMEROPS anchor skipped: {e}")


if __name__ == "__main__":
    main()
