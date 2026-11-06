#!/usr/bin/env python
"""Thermal adaptation as a CONTINUOUS function of distance from the catalytic serine.

The pocket-versus-control contrast is binary and depends on a chosen boundary. This instead bins every
residue by its distance from the catalytic Ser and asks, for each shell independently, how strongly that
shell tracks growth temperature within clades. Two readouts are computed on the same residues:

  attribution   mean of ESM-2's exact per-residue contribution to predicted temperature (as in stage 14)
  composition   mean residue volume / IVYWREL / polar fraction (as in stage 11)

If the pocket is shielded, both should rise from near zero at the active site and plateau further out.
A smooth gradient is far harder to explain as an artifact than a two-group difference.

  python scripts/15_distance_profile.py --config config.yaml
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
from serprot.structure import KD, RESVOL, neighbor_count, read_pdb  # noqa: E402

# tuned to real serine-protease geometry: median shell counts are ~11 residues within 6 A,
# ~22 within 8 A, ~39 within 10 A, so these bins each hold enough residues to average over
DEFAULT_EDGES = [0, 5, 8, 11, 14, 18, 24, 32, 999]
IVYWREL = set("IVYWREL")
POLAR = set("STNQHCY")


EDGES = DEFAULT_EDGES


def bin_label(lo, hi):
    return f"{lo:g}-{hi:g}" if hi < 900 else f">{lo:g}"


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--structures-dir", default=None, help="structure directory relative to paths.root (default: paths.structures); point this at data/structures_esmfold for the ESMFold replication")
    ap.add_argument("--tag", default="", help='suffix added to output filenames (e.g. _esmfold) so runs do not overwrite each other')
    ap.add_argument("--min-residues", type=int, default=5, help="min residues in a shell to use that protein")
    ap.add_argument("--no-attribution", action="store_true", help="composition only; skips the GPU pass")
    ap.add_argument("--no-burial-control", action="store_true",
                    help="skip the burial-adjusted pass (it doubles the statistics runtime)")
    ap.add_argument("--edges", default=",".join(map(str, DEFAULT_EDGES)),
                    help="distance-shell boundaries in Angstrom, comma separated")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_all(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    global EDGES
    EDGES = [float(e) for e in args.edges.split(",")]
    print("distance shells (A):", " ".join(bin_label(a, b) for a, b in zip(EDGES[:-1], EDGES[1:])))

    full = prepare_inputs(load_dataset(cfg), cfg["max_len"])
    sdir = (Path(cfg["paths"]["root"]).expanduser() / args.structures_dir
            if args.structures_dir else Path(rpath(cfg, "structures", "x")).parent)
    print(f"structures: {sdir}")
    tg = set(pd.read_csv(rpath(cfg, "processed", "structure_targets.tsv"), sep="\t")["accession"])
    df = full[full["accession"].isin(tg) & full["nuc_pos"].notna() & full["ogt"].notna()].reset_index(drop=True)
    print(f"{len(df)} candidate proteins")

    W, model, tok, dev = None, None, None, None
    if not args.no_attribution:
        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import StandardScaler
        slug = re.sub(r"[^A-Za-z0-9]+", "_", str(cfg["esm_model"]).rstrip("/").split("/")[-1]).strip("_")
        cache = Path(rpath(cfg, "processed", f"esm_embeddings_{slug}.npy"))
        if not cache.exists():
            raise SystemExit(f"need {cache}; run stage 4 or pass --no-attribution")
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
        print(f"attribution enabled; folds {sorted(W)}; device {dev}")

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
            ri, pt, _how = anchor_point(st, r["nuc_pos"])
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
            for row, (j, r, st, dist, _nc) in enumerate(prepared):
                if r["fold_cv"] not in W:
                    continue
                wt = torch.as_tensor(W[r["fold_cv"]], dtype=torch.float32, device=dev)
                n_tok = int(enc["attention_mask"][row].sum())
                S[j] = (H[row] @ wt).cpu().numpy()[1 : n_tok - 1]

        for j, r, st, dist, nc in prepared:
            s_res = S.get(j)
            for lo, hi in zip(EDGES[:-1], EDGES[1:]):
                sel = np.flatnonzero((dist >= lo) & (dist < hi))
                if len(sel) < args.min_residues:
                    continue
                aa = st.aa[sel]
                rec = {"accession": r["accession"], "family": r["family"],
                       "cluster_pairs": r["cluster_pairs"], "ogt": float(r["ogt"]),
                       "shell": bin_label(lo, hi), "shell_lo": lo, "n_res": len(sel),
                       "resvol": float(np.mean([RESVOL.get(a, 135.0) for a in aa])),
                       "f_ivywrel": float(np.mean([a in IVYWREL for a in aa])),
                       "f_polar": float(np.mean([a in POLAR for a in aa])),
                       "burial": float(np.mean(nc[sel])),
                       "hydropathy": float(np.mean([KD.get(a, 0.0) for a in aa]))}
                if s_res is not None and len(s_res):
                    pos = to_sequence_positions(st, sel, r["crop_off"], len(s_res))
                    if len(pos) >= args.min_residues:
                        rec["attribution"] = float(s_res[pos].mean())
                rows.append(rec)
        if (bi // args.batch) % 100 == 0:
            done = bi + len(idx)
            print(f"  {done}/{len(order)}  {done / max(time.time() - t0, 1):.0f}/s", flush=True)

    d = pd.DataFrame(rows)
    print(f"\n{d['accession'].nunique()} proteins, {len(d)} protein-shell rows; skipped {skipped}")
    out = rpath(cfg, "runs", "analysis", f"distance_profile{args.tag}.tsv")
    d.to_csv(str(out).replace(".tsv", "_raw.tsv"), sep="\t", index=False)

    measures = [m for m in ["attribution", "resvol", "f_ivywrel", "f_polar", "hydropathy"] if m in d]
    res = []
    passes = [("raw", False)] if args.no_burial_control else [("raw", False), ("burial_adj", True)]
    for adj_name, use_burial in passes:
        note = ("" if not use_burial else
                "   [shell burial partialled out: thermal adaptation is known to favour the surface, "
                "so a distance gradient could be an exposure gradient]")
        for meas in measures:
            print(f"\n===== {meas} / {adj_name}: within-clade slope on OGT per +10 C, "
                  f"BY DISTANCE FROM CATALYTIC Ser ====={note}")
            print(f"{'shell (A)':>10} {'proteins':>9} {'slope':>10} {'95% CI':>22} {'p_perm':>8} {'burial':>8}")
            for lo, hi in zip(EDGES[:-1], EDGES[1:]):
                g = d[(d["shell_lo"] == lo) & d[meas].notna()]
                if g["cluster_pairs"].nunique() < 20:
                    continue
                cl, codes = np.unique(g["cluster_pairs"].values, return_inverse=True)
                k = len(cl)
                v, x = g[meas].values.astype(float), g["ogt"].values.astype(float)
                z = g["burial"].values.astype(float) if use_burial else None
                sl = slope(v, x, codes, k, z)
                bs = clade_bootstrap(lambda i, rc: slope(v[i], x[i], rc, k, None if z is None else z[i]),
                                     codes, k, args.n_boot, rng)
                lo_ci, hi_ci = np.percentile(bs, [2.5, 97.5]) if len(bs) > 10 else (np.nan, np.nan)
                null = np.array([slope(v, permute_within_clades(x, codes, k, rng), codes, k, z)
                                 for _ in range(args.n_perm)])
                null = null[np.isfinite(null)]
                p = float((np.abs(null) >= abs(sl)).mean()) if len(null) else np.nan
                print(f"{bin_label(lo, hi):>10} {len(g):>9} {sl:>10.4f} "
                      f"{f'[{lo_ci:+.4f}, {hi_ci:+.4f}]':>22} {p:>8.4f} {g['burial'].mean():>8.1f}")
                res.append({"measure": meas, "adjustment": adj_name, "shell": bin_label(lo, hi),
                            "shell_lo": lo, "n": len(g), "clades": k, "slope": sl, "ci_lo": lo_ci,
                            "ci_hi": hi_ci, "p_perm": p, "mean_burial": float(g["burial"].mean())})
    pd.DataFrame(res).to_csv(out, sep="\t", index=False)
    print(f"\nwrote {out}")
    print("\nRead the innermost shells against the outer ones: a signal that climbs with distance is")
    print("shielding; a flat profile would mean the pocket/control split was arbitrary.")
    print("Then compare raw with burial_adj. If the gradient SURVIVES burial adjustment it is specific to")
    print("the active site. If it collapses, the finding is the already-known surface preference of")
    print("thermal adaptation, and the active-site claim cannot be made.")
    print("The 'burial' column shows mean neighbour count per shell -- check whether it trends with distance.")


if __name__ == "__main__":
    main()
