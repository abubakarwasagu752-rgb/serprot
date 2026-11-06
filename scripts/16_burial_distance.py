#!/usr/bin/env python
"""Separate two entangled explanations: is thermal adaptation excluded from the ACTIVE SITE, or just
concentrated on the SURFACE?

Burial falls steadily with distance from the catalytic Ser, so a distance gradient and an exposure
gradient look identical in one dimension. This bins every residue JOINTLY by distance and by burial
(per-protein terciles of neighbour count, so protein size and compactness cannot drive the split),
and measures the within-clade temperature slope in each cell.

  Read the result DOWN each burial column:
    slope still rises with distance inside a fixed burial stratum -> the effect is specific to the
      active site, over and above exposure
    slope flat within strata but rising ACROSS them -> the effect is exposure, and the active-site
      framing cannot be supported

  python scripts/16_burial_distance.py --config config.yaml
"""
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serprot.common import base_parser, load_config, load_dataset, prepare_inputs, rpath, seed_all  # noqa: E402
from serprot.pockets import anchor_point, to_sequence_positions  # noqa: E402
from serprot.stats import clade_bootstrap, permute_within_clades, slope  # noqa: E402
from serprot.structure import RESVOL, neighbor_count, read_pdb  # noqa: E402

IVYWREL = set("IVYWREL")
POLAR = set("STNQHCY")


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--dist-edges", default="0,10,18,28,999")
    ap.add_argument("--n-burial", type=int, default=3, help="per-protein burial strata (terciles by default)")
    ap.add_argument("--min-residues", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=800)
    ap.add_argument("--n-perm", type=int, default=800)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--structures-dir", default=None, help="structure directory relative to paths.root (default: paths.structures); point this at data/structures_esmfold for the ESMFold replication")
    ap.add_argument("--tag", default="", help='suffix added to output filenames (e.g. _esmfold) so runs do not overwrite each other')
    ap.add_argument("--no-attribution", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_all(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    D = [float(x) for x in args.dist_edges.split(",")]
    dlab = [f"{a:g}-{b:g}" if b < 900 else f">{a:g}" for a, b in zip(D[:-1], D[1:])]

    full = prepare_inputs(load_dataset(cfg), cfg["max_len"])
    sdir = (Path(cfg["paths"]["root"]).expanduser() / args.structures_dir
            if args.structures_dir else Path(rpath(cfg, "structures", "x")).parent)
    print(f"structures: {sdir}")
    tg = set(pd.read_csv(rpath(cfg, "processed", "structure_targets.tsv"), sep="\t")["accession"])
    df = full[full["accession"].isin(tg) & full["nuc_pos"].notna() & full["ogt"].notna()].reset_index(drop=True)
    print(f"{len(df)} candidate proteins; distance bins {dlab}; {args.n_burial} burial strata per protein")

    W = model = tok = dev = None
    if not args.no_attribution:
        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import StandardScaler
        slug = re.sub(r"[^A-Za-z0-9]+", "_", str(cfg["esm_model"]).rstrip("/").split("/")[-1]).strip("_")
        cache = Path(rpath(cfg, "processed", f"esm_embeddings_{slug}.npy"))
        if not cache.exists():
            raise SystemExit(f"need {cache} (stage 4) or pass --no-attribution")
        E_all = np.load(cache)
        y_all, f_all = full["ogt"].values.astype(float), full["fold_cv"].values
        W = {}
        for k in np.unique(f_all):
            tr = (f_all != k) & np.isfinite(y_all)
            if tr.sum() < 50:
                continue
            sc = StandardScaler().fit(E_all[tr])
            m = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(sc.transform(E_all[tr]), y_all[tr])
            W[k] = m.coef_ / sc.scale_
        import torch
        from transformers import EsmModel, EsmTokenizer
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tok = EsmTokenizer.from_pretrained(cfg["esm_model"])
        model = EsmModel.from_pretrained(cfg["esm_model"], add_pooling_layer=False).to(dev).eval()
        print(f"attribution enabled on {dev}")

    rows, skipped, t0 = [], {"no_structure": 0, "no_anchor": 0}, time.time()
    order = np.argsort([len(s) for s in df["seq_in"]])
    for bi in range(0, len(order), args.batch):
        idx = order[bi : bi + args.batch]
        prepared = []
        for j in idx:
            r = df.iloc[j]
            f = sdir / f"{r['accession']}.pdb.gz"
            if not f.exists():
                skipped["no_structure"] += 1
                continue
            try:
                st = read_pdb(f)
            except Exception:  # noqa: BLE001
                skipped["no_structure"] += 1
                continue
            ri, pt, _ = anchor_point(st, r["nuc_pos"])
            if ri is None:
                skipped["no_anchor"] += 1
                continue
            prepared.append((j, r, st, st.min_dist_to(pt), neighbor_count(st)))
        if not prepared:
            continue
        S = {}
        if W is not None:
            import torch
            enc = tok([p[1]["seq_in"] for p in prepared], padding=True, return_tensors="pt").to(dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                H = model(**enc).last_hidden_state.float()
            for row, (j, r, st, dist, nc) in enumerate(prepared):
                if r["fold_cv"] in W:
                    wt = torch.as_tensor(W[r["fold_cv"]], dtype=torch.float32, device=dev)
                    n_tok = int(enc["attention_mask"][row].sum())
                    S[j] = (H[row] @ wt).cpu().numpy()[1 : n_tok - 1]

        for j, r, st, dist, nc in prepared:
            # burial strata are PER PROTEIN, so compactness differences between proteins cannot
            # masquerade as an exposure effect
            qs = np.quantile(nc, np.linspace(0, 1, args.n_burial + 1)[1:-1])
            strat = np.digitize(nc, qs)
            s_res = S.get(j)
            for di, (lo, hi) in enumerate(zip(D[:-1], D[1:])):
                for b in range(args.n_burial):
                    sel = np.flatnonzero((dist >= lo) & (dist < hi) & (strat == b))
                    if len(sel) < args.min_residues:
                        continue
                    aa = st.aa[sel]
                    rec = {"accession": r["accession"], "family": r["family"],
                           "cluster_pairs": r["cluster_pairs"], "ogt": float(r["ogt"]),
                           "dist_bin": dlab[di], "dist_i": di, "burial_stratum": b, "n_res": len(sel),
                           "mean_burial": float(nc[sel].mean()),
                           "resvol": float(np.mean([RESVOL.get(a, 135.0) for a in aa])),
                           "f_ivywrel": float(np.mean([a in IVYWREL for a in aa])),
                           "f_polar": float(np.mean([a in POLAR for a in aa]))}
                    if s_res is not None and len(s_res):
                        pos = to_sequence_positions(st, sel, r["crop_off"], len(s_res))
                        if len(pos) >= args.min_residues:
                            rec["attribution"] = float(s_res[pos].mean())
                    rows.append(rec)
        if (bi // args.batch) % 150 == 0:
            done = bi + len(idx)
            print(f"  {done}/{len(order)}  {done / max(time.time() - t0, 1):.0f}/s", flush=True)

    d = pd.DataFrame(rows)
    print(f"\n{d['accession'].nunique()} proteins, {len(d)} cells; skipped {skipped}")
    out = rpath(cfg, "runs", "analysis", f"burial_distance{args.tag}.tsv")
    d.to_csv(str(out).replace(".tsv", "_raw.tsv"), sep="\t", index=False)

    bl = {0: "most exposed", args.n_burial - 1: "most buried"}
    res = []
    for meas in [m for m in ["attribution", "f_ivywrel", "f_polar", "resvol"] if m in d]:
        print(f"\n===== {meas}: within-clade slope on OGT per +10 C =====")
        print("rows = distance from catalytic Ser; columns = burial stratum WITHIN each protein")
        labels = [f"stratum {b} ({bl.get(b, 'mid')})" for b in range(args.n_burial)]
        hdr = "".join(f"{lab:>26}" for lab in labels)
        print(f"{'distance':>10}{hdr}")
        for di, lab in enumerate(dlab):
            cells = []
            for b in range(args.n_burial):
                g = d[(d["dist_i"] == di) & (d["burial_stratum"] == b) & d[meas].notna()]
                if g["cluster_pairs"].nunique() < 20:
                    cells.append("--")
                    continue
                cl, codes = np.unique(g["cluster_pairs"].values, return_inverse=True)
                k = len(cl)
                v, x = g[meas].values.astype(float), g["ogt"].values.astype(float)
                sl = slope(v, x, codes, k)
                null = np.array([slope(v, permute_within_clades(x, codes, k, rng), codes, k)
                                 for _ in range(args.n_perm)])
                null = null[np.isfinite(null)]
                p = float((np.abs(null) >= abs(sl)).mean()) if len(null) else np.nan
                bs = clade_bootstrap(lambda i, rc: slope(v[i], x[i], rc, k), codes, k, args.n_boot, rng)
                lo_ci, hi_ci = np.percentile(bs, [2.5, 97.5]) if len(bs) > 10 else (np.nan, np.nan)
                cells.append(f"{sl:+.4f} p={p:.3f}")
                res.append({"measure": meas, "dist_bin": lab, "dist_i": di, "burial_stratum": b,
                            "slope": sl, "ci_lo": lo_ci, "ci_hi": hi_ci, "p_perm": p, "n": len(g),
                            "clades": k, "mean_burial": float(g["mean_burial"].mean())})
            print(f"{lab:>10}" + "".join(f"{c:>26}" for c in cells))
        print(f"{'burial:':>10}" + "".join(
            f"{d[(d.burial_stratum == b)]['mean_burial'].mean():>26.1f}" for b in range(args.n_burial)))
    pd.DataFrame(res).to_csv(out, sep="\t", index=False)
    print(f"\nwrote {out}")
    print("\nDOWN a column = distance effect at fixed exposure (the active-site claim).")
    print("ACROSS a row  = exposure effect at fixed distance (the known surface preference).")
    print("Both can be real; what matters is whether the column trend survives at all.")


if __name__ == "__main__":
    main()
