#!/usr/bin/env python
"""Does the catalytic neighbourhood resist thermal adaptation in its FLEXIBILITY as well as its composition?

Everything so far measured what the pocket is MADE OF. Kulathunga & Potoyan (bioRxiv 2025.03.03.641325)
measured something else on four structures: how much it MOVES, as C-alpha fluctuation. Composition and
dynamics are different observables, and a pocket that is compositionally frozen could still be
dynamically tuned, so the composition result does not settle their question.

This closes that gap at this project's scale. Per-residue mean-square fluctuation comes from a Gaussian
Network Model built on the C-alpha coordinates -- one eigendecomposition per protein, the standard cheap
stand-in for RMSF, and the same quantity B-factors report. It is then put through EXACTLY the two tests
the composition descriptors went through:

  1. pocket vs the size- and burial-matched control patch on the same protein (stage 12's contrast)
  2. the continuous profile against distance from the catalytic Ser (stage 15's gradient)

  Read the result:
    pocket slope ~ 0 while the control slope is positive  -> the exclusion extends to dynamics, and the
      trade-off claim covers Kulathunga & Potoyan's observable too
    pocket and control slopes alike                       -> thermal adaptation DOES tune pocket dynamics,
      and the right claim is narrower: composition is spared, motion is not

Fluctuations are normalised by each protein's own mean, so protein size and overall compactness cannot
drive the comparison; the unnormalised version is reported beside it.

  python scripts/18_flexibility.py --config config.yaml --workers 16
  python scripts/18_flexibility.py --config config.yaml --structures-dir data/structures_esmfold --tag _esmfold
"""
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serprot.common import base_parser, load_config, load_dataset, prepare_inputs, rpath, seed_all  # noqa: E402
from serprot.pockets import anchor_point, regions_for  # noqa: E402
from serprot.stats import clade_bootstrap, permute_within_clades, slope, within_sd  # noqa: E402
from serprot.structure import read_pdb  # noqa: E402


def gnm_msf(ca: np.ndarray, cutoff: float):
    """Gaussian Network Model mean-square fluctuation per residue.

    The Kirchhoff matrix of the C-alpha contact graph is inverted on its non-zero modes; the diagonal of
    that pseudo-inverse is the MSF. Returns (msf [R] with NaN where there is no C-alpha, n_zero_modes).
    A connected fold gives exactly one zero mode; more than one means the contact graph fell apart
    (a chain break or a detached domain), and those proteins are dropped at analysis time because
    fluctuations are not comparable across disconnected components.
    """
    from scipy.spatial import cKDTree

    ok = ~np.isnan(ca[:, 0])
    X = np.asarray(ca[ok], dtype=np.float64)
    n = len(X)
    if n < 20:
        return None, 0
    g = np.zeros((n, n), dtype=np.float64)
    pairs = cKDTree(X).query_pairs(cutoff, output_type="ndarray")
    if not len(pairs):
        return None, 0
    g[pairs[:, 0], pairs[:, 1]] = -1.0
    g = g + g.T
    np.fill_diagonal(g, -g.sum(axis=1))
    w, v = np.linalg.eigh(g)
    tol = 1e-8 * max(float(w.max()), 1.0)
    keep = w > tol
    n_zero = int((~keep).sum())
    if not keep.any():
        return None, n_zero
    msf_ok = ((v[:, keep] ** 2) / w[keep]).sum(axis=1)
    out = np.full(len(ca), np.nan, dtype=np.float64)
    out[ok] = msf_ok
    return out, n_zero


def _one(task):
    """One protein: fold-independent, so this runs in a worker process."""
    (acc, nuc_pos, sdir, radius, cutoff, edges, max_res) = task
    f = Path(sdir) / f"{acc}.pdb.gz"
    if not f.exists():
        return {"accession": acc, "status": "no_structure"}
    try:
        st = read_pdb(f)
    except Exception as e:  # noqa: BLE001
        return {"accession": acc, "status": f"unreadable:{type(e).__name__}"}
    if st is None:
        return {"accession": acc, "status": "no_structure"}
    if len(st) > max_res:
        return {"accession": acc, "status": "too_long", "n_res": len(st)}
    reg = regions_for(st, int(nuc_pos), acc, radius=radius)
    if reg is None:
        return {"accession": acc, "status": "no_anchor"}
    msf, n_zero = gnm_msf(st.ca, cutoff)
    if msf is None:
        return {"accession": acc, "status": "gnm_failed"}
    mean_msf = float(np.nanmean(msf))
    if not np.isfinite(mean_msf) or mean_msf <= 0:
        return {"accession": acc, "status": "gnm_degenerate"}
    rel = msf / mean_msf

    pk = reg["pocket"]
    row = {"accession": acc, "status": "ok", "n_res": len(st), "n_zero_modes": n_zero,
           "mean_msf": mean_msf, "how": reg["how"],
           "pk_flex": float(np.nanmean(rel[pk])), "pk_flex_raw": float(np.nanmean(msf[pk])),
           "pk_plddt": float(np.nanmean(st.plddt[pk])), "pk_n": int(len(pk))}
    ctrls = reg["controls"]
    if ctrls:
        row["ct_flex"] = float(np.mean([np.nanmean(rel[c]) for c in ctrls]))
        row["ct_flex_raw"] = float(np.mean([np.nanmean(msf[c]) for c in ctrls]))
        row["ct_n_patches"] = len(ctrls)

    # continuous profile: the same shells stage 15 used, so the two gradients are directly comparable
    _ri, pt, _how = anchor_point(st, int(nuc_pos))
    dist = st.min_dist_to(pt)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        sel = np.flatnonzero((dist >= lo) & (dist < hi) & np.isfinite(rel))
        if len(sel) >= 4:
            row[f"shell{i}"] = float(rel[sel].mean())
            row[f"shell{i}_n"] = int(len(sel))
    return row


def report(df, ycol, ctcol, label, n_boot, n_perm, rng, standardize=False):
    """Within-clade slope on OGT for a pocket column and its matched control, plus the difference."""
    sub = df[df[ycol].notna() & df[ctcol].notna() & df["ogt"].notna()]
    if sub["cluster_pairs"].nunique() < 20:
        print(f"{label}: too few clades ({sub['cluster_pairs'].nunique()})")
        return None
    _cl, codes = np.unique(sub["cluster_pairs"].values, return_inverse=True)
    k = int(codes.max()) + 1
    x = sub["ogt"].values.astype(float)
    yp = sub[ycol].values.astype(float)
    yc = sub[ctcol].values.astype(float)
    sd_p, sd_c = within_sd(yp, codes, k), within_sd(yc, codes, k)
    if standardize:
        yp = yp / sd_p if np.isfinite(sd_p) and sd_p > 0 else yp
        yc = yc / sd_c if np.isfinite(sd_c) and sd_c > 0 else yc
    sp, sc = slope(yp, x, codes, k), slope(yc, x, codes, k)
    diff = sp - sc
    bs = clade_bootstrap(lambda i, rc: slope(yp[i], x[i], rc, k) - slope(yc[i], x[i], rc, k),
                         codes, k, n_boot, rng)
    lo, hi = (np.percentile(bs, [2.5, 97.5]) if len(bs) > 10 else (np.nan, np.nan))
    null = []
    for _ in range(n_perm):
        xs = permute_within_clades(x, codes, k, rng)
        null.append(slope(yp, xs, codes, k) - slope(yc, xs, codes, k))
    null = np.array([v for v in null if np.isfinite(v)])
    p = float((np.abs(null) >= abs(diff)).mean()) if len(null) else np.nan
    print(f"{label:>26}  pocket {sp:+.5f}   control {sc:+.5f}   difference {diff:+.5f} "
          f"[{lo:+.5f}, {hi:+.5f}]  p={p:.4f}   n={len(sub)}  clades={k}")
    return {"measure": label, "slope_pocket": sp, "slope_control": sc, "difference": diff,
            "difference_lo": lo, "difference_hi": hi, "p_perm": p, "n": len(sub), "clades": k,
            "within_sd_pocket": sd_p, "within_sd_control": sd_c}


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--radius", type=float, default=10.0, help="pocket radius in A (match the stage 12 run)")
    ap.add_argument("--cutoff", type=float, default=7.3,
                    help="GNM contact cutoff in A; 7.3 is the value fitted against crystallographic B-factors")
    ap.add_argument("--dist-edges", default="0,5,8,11,14,18,24,32,999",
                    help="shell edges for the continuous profile (stage 15 defaults)")
    ap.add_argument("--min-plddt", type=float, default=70.0)
    ap.add_argument("--max-residues", type=int, default=2200,
                    help="skip very long chains; the eigendecomposition is cubic in chain length")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=0, help="0 = SLURM_CPUS_PER_TASK, or all cores")
    ap.add_argument("--structures-dir", default=None,
                    help="relative to paths.root; point at data/structures_esmfold for the replication")
    ap.add_argument("--tag", default="", help="suffix for output filenames, e.g. _esmfold")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_all(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    edges = [float(v) for v in args.dist_edges.split(",")]
    labels = [f"{a:g}-{b:g}" if b < 900 else f">{a:g}" for a, b in zip(edges[:-1], edges[1:])]

    full = prepare_inputs(load_dataset(cfg), cfg["max_len"])
    sdir = (Path(cfg["paths"]["root"]).expanduser() / args.structures_dir
            if args.structures_dir else Path(rpath(cfg, "structures", "x")).parent)
    tg = pd.read_csv(rpath(cfg, "processed", "structure_targets.tsv"), sep="\t")
    df = full[full["accession"].isin(set(tg["accession"])) & full["nuc_pos"].notna()].reset_index(drop=True)
    print(f"structures: {sdir}")
    print(f"{len(df)} candidate proteins; GNM cutoff {args.cutoff} A; pocket radius {args.radius} A")

    tasks = [(r["accession"], r["nuc_pos"], str(sdir), args.radius, args.cutoff, edges, args.max_residues)
             for _, r in df.iterrows()]
    nw = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    nw = max(1, min(nw, len(tasks)))
    print(f"computing fluctuations on {nw} workers", flush=True)
    t0, rows = time.time(), []
    if nw == 1:
        for i, t in enumerate(tasks):
            rows.append(_one(t))
            if i % 500 == 0:
                print(f"  {i}/{len(tasks)}  {(i + 1) / max(time.time() - t0, 1):.1f}/s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=nw) as ex:
            for i, r in enumerate(ex.map(_one, tasks, chunksize=16)):
                rows.append(r)
                if i % 1000 == 0:
                    print(f"  {i}/{len(tasks)}  {(i + 1) / max(time.time() - t0, 1):.1f}/s", flush=True)
    print(f"fluctuations done in {(time.time() - t0) / 60:.1f} min")

    d = pd.DataFrame(rows)
    print("\nstatus:")
    print(d["status"].value_counts().to_string())
    raw_out = rpath(cfg, "runs", "analysis", f"flexibility{args.tag}_raw.tsv")
    d.to_csv(raw_out, sep="\t", index=False)
    print(f"wrote {raw_out}")

    d = d[d["status"] == "ok"].merge(tg[["accession", "family", "cluster_pairs", "ogt"]],
                                     on="accession", how="inner")
    n_before = len(d)
    d = d[d["n_zero_modes"] <= 1]
    print(f"dropped {n_before - len(d)} proteins whose contact graph was disconnected")
    n_before = len(d)
    d = d[d["pk_plddt"] >= args.min_plddt]
    print(f"dropped {n_before - len(d)} proteins with pocket pLDDT < {args.min_plddt}")
    good = d.groupby("cluster_pairs")["ogt"].transform("size") >= 2
    spread = d.groupby("cluster_pairs")["ogt"].transform(lambda s: s.max() - s.min()) > 0
    d = d[good & spread]
    n_hot = int((d["ogt"] >= 50).sum())
    print(f"analysing {len(d)} proteins in {d['cluster_pairs'].nunique()} clades ({n_hot} thermophilic)")

    res = []
    print("\n===== 1. pocket vs matched control: within-clade slope of FLEXIBILITY on OGT, per +10 C =====")
    print("   normalised = each protein's fluctuations divided by its own mean")
    for fam in ["all"] + sorted(d["family"].unique()):
        sub = d if fam == "all" else d[d["family"] == fam]
        if sub["cluster_pairs"].nunique() < 20:
            continue
        print(f"\n-- {fam} --")
        for ycol, ctcol, name in [("pk_flex", "ct_flex", "normalised"),
                                  ("pk_flex_raw", "ct_flex_raw", "unnormalised")]:
            if ycol not in sub or ctcol not in sub:
                continue
            r = report(sub, ycol, ctcol, name, args.n_boot, args.n_perm, rng)
            if r:
                res.append({"family": fam, "pass": "raw", **r})
        r = report(sub, "pk_flex", "ct_flex", "normalised, standardized",
                   args.n_boot, args.n_perm, rng, standardize=True)
        if r:
            res.append({"family": fam, "pass": "standardized", **r})

    print("\n===== 2. continuous profile: flexibility by distance from the catalytic Ser =====")
    print("   compare the shape against stage 15's attribution profile, which used the same shells")
    prof = []
    print(f"{'shell (A)':>12}{'mean rel. flex':>18}{'slope / +10 C':>18}{'p':>10}{'n':>10}{'clades':>9}")
    for i, lab in enumerate(labels):
        col = f"shell{i}"
        if col not in d:
            continue
        sub = d[d[col].notna() & d["ogt"].notna()]
        if sub["cluster_pairs"].nunique() < 20:
            continue
        _cl, codes = np.unique(sub["cluster_pairs"].values, return_inverse=True)
        k = int(codes.max()) + 1
        y, x = sub[col].values.astype(float), sub["ogt"].values.astype(float)
        sl = slope(y, x, codes, k)
        null = np.array([slope(y, permute_within_clades(x, codes, k, rng), codes, k)
                         for _ in range(args.n_perm)])
        null = null[np.isfinite(null)]
        p = float((np.abs(null) >= abs(sl)).mean()) if len(null) else np.nan
        bs = clade_bootstrap(lambda idx, rc: slope(y[idx], x[idx], rc, k), codes, k, args.n_boot, rng)
        lo, hi = (np.percentile(bs, [2.5, 97.5]) if len(bs) > 10 else (np.nan, np.nan))
        mean_rel = float(y.mean())
        print(f"{lab:>12}{mean_rel:>18.4f}{sl:>+18.5f}{p:>10.4f}{len(sub):>10}{k:>9}")
        prof.append({"shell": lab, "shell_i": i, "mean_rel_flex": mean_rel, "slope": sl,
                     "ci_lo": lo, "ci_hi": hi, "p_perm": p, "n": len(sub), "clades": k})

    out = rpath(cfg, "runs", "analysis", f"flexibility{args.tag}.tsv")
    pd.DataFrame(res).to_csv(out, sep="\t", index=False)
    pout = rpath(cfg, "runs", "analysis", f"flexibility_profile{args.tag}.tsv")
    pd.DataFrame(prof).to_csv(pout, sep="\t", index=False)
    print(f"\nwrote {out}")
    print(f"wrote {pout}")
    print("\nThe honest reading: a GNM fluctuation is a harmonic approximation from one predicted")
    print("conformer, not an MD trajectory. It answers whether the pocket's RELATIVE mobility tracks")
    print("growth temperature; it cannot speak to anharmonic motion, unfolding, or solvent effects.")


if __name__ == "__main__":
    main()
