#!/usr/bin/env python
"""Describe each protein's substrate-binding region, and a matched control region, from its AlphaFold model.

For every protein in structure_targets.tsv:
  pocket   = residues with a heavy atom within --radius A of the catalytic Ser OG
  controls = --n-controls patches elsewhere in the same protein (centre > --control-min-dist A from the OG,
             burial matched to the pocket centre), each grown to the same residue count as the pocket
Descriptors are computed identically for both, so the analysis can ask whether temperature moves the
pocket differently from the rest of the protein rather than just moving everything.

  python scripts/11_pocket_descriptors.py --config config.yaml --workers 16
  python scripts/11_pocket_descriptors.py --config config.yaml --shard $SLURM_ARRAY_TASK_ID --n-shards 8
"""
import hashlib
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serprot.common import base_parser, load_config, rpath  # noqa: E402
from serprot.structure import AROMATIC, CHARGED_NEG, CHARGED_POS, KD, POLAR, RESVOL, SMALL  # noqa: E402
from serprot.structure import neighbor_count, read_pdb, sasa  # noqa: E402

SHELLS = (6.0, 8.0, 10.0)


def set_descriptors(st, idx, sa, nc, prefix: str) -> dict:
    """Composition / geometry / exposure of one residue set."""
    if len(idx) == 0:
        return {}
    aa = st.aa[idx]
    ca = st.ca[idx]
    ca = ca[~np.isnan(ca[:, 0])]
    cen = ca.mean(0) if len(ca) else np.full(3, np.nan)
    d = {
        f"{prefix}_n": len(idx),
        f"{prefix}_hydropathy": float(np.mean([KD.get(a, 0.0) for a in aa])),
        f"{prefix}_f_pos": float(np.mean([a in CHARGED_POS for a in aa])),
        f"{prefix}_f_neg": float(np.mean([a in CHARGED_NEG for a in aa])),
        f"{prefix}_net_charge": float(sum(a in CHARGED_POS for a in aa) - sum(a in CHARGED_NEG for a in aa)),
        f"{prefix}_f_aromatic": float(np.mean([a in AROMATIC for a in aa])),
        f"{prefix}_f_polar": float(np.mean([a in POLAR for a in aa])),
        f"{prefix}_f_small": float(np.mean([a in SMALL for a in aa])),
        f"{prefix}_f_gly": float(np.mean(aa == "G")),
        f"{prefix}_f_pro": float(np.mean(aa == "P")),
        f"{prefix}_resvol": float(np.mean([RESVOL.get(a, 135.0) for a in aa])),
        f"{prefix}_f_ivywrel": float(np.mean([a in set("IVYWREL") for a in aa])),   # classic thermophily index
        f"{prefix}_sasa_mean": float(np.nanmean(sa[idx])),
        f"{prefix}_sasa_total": float(np.nansum(sa[idx])),
        f"{prefix}_burial": float(np.mean(nc[idx])),
        f"{prefix}_plddt": float(np.mean(st.plddt[idx])),
        f"{prefix}_rgyr": float(np.sqrt(((ca - cen) ** 2).sum(1).mean())) if len(ca) > 1 else np.nan,
    }
    return d


def cavity_volume(st, og, radius=12.0, spacing=0.8, probe=1.4, enclosure=55) -> tuple[float, float]:
    """Empty space around the catalytic Ser, connected to it. Returns (open_vol, enclosed_vol) in A^3.

    `open_vol` counts every empty grid point inside the sphere, so for a surface active site it includes
    bulk solvent. `enclosed_vol` keeps only points with at least `enclosure` heavy atoms within 8 A, which
    is what separates a genuine cleft from open solvent; prefer it as the pocket-size measure.
    """
    from scipy import ndimage
    from scipy.spatial import cKDTree

    g = np.arange(-radius, radius + spacing, spacing)
    gx, gy, gz = np.meshgrid(g, g, g, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1) + og
    inside = np.linalg.norm(pts - og, axis=1) <= radius
    tree = cKDTree(st.coords)
    empty = tree.query_ball_point(pts, r=1.8 + probe, workers=-1, return_length=True) == 0
    env = tree.query_ball_point(pts, r=8.0, workers=-1, return_length=True)

    def connected_volume(mask):
        lab, _ = ndimage.label(mask.reshape(gx.shape))
        c = np.array(gx.shape) // 2
        rr = max(int(round(3.0 / spacing)), 1)
        sl = tuple(slice(max(ci - rr, 0), ci + rr + 1) for ci in c)
        ids = np.unique(lab[sl])
        ids = ids[ids > 0]
        if len(ids) == 0:
            return 0.0
        return float(max((lab == i).sum() for i in ids) * spacing**3)

    return connected_volume(empty & inside), connected_volume(empty & inside & (env >= enclosure))


def one_protein(args) -> dict | None:
    acc, nuc_pos, sdir, radius, n_ctrl, ctrl_min_dist, n_sphere, enclosure = args
    f = Path(sdir) / f"{acc}.pdb.gz"
    if not f.exists():
        f = Path(sdir) / f"{acc}.pdb"
        if not f.exists():
            return None
    try:
        st = read_pdb(f)
    except Exception:  # noqa: BLE001
        return None
    if st is None or len(st) < 50:
        return None
    ri = st.res_index(int(nuc_pos))
    if ri is None:
        return {"accession": acc, "status": "nuc_pos_not_in_model"}
    og = st.atom_coord(ri, "OG")
    if og is None:                       # not a Ser in the model, or side chain missing
        og = st.ca[ri]
        if np.isnan(og).any():
            return {"accession": acc, "status": "no_coords_at_nuc_pos"}
        anchor = "CA_fallback"
    else:
        anchor = "OG"

    dist = st.min_dist_to(og)
    nc = neighbor_count(st)
    pocket = {s: np.flatnonzero(dist <= s) for s in SHELLS}
    main = pocket[radius]
    if len(main) < 5:
        return {"accession": acc, "status": "pocket_too_small"}

    # control patches: centres far from the pocket, burial-matched, grown to the pocket's size
    rng = np.random.default_rng(int(hashlib.sha1(acc.encode()).hexdigest()[:8], 16))
    far = np.flatnonzero(dist > ctrl_min_dist)
    target_burial = nc[ri]
    ctrl_sets = []
    if len(far):
        order = far[np.argsort(np.abs(nc[far] - target_burial))]
        pool = order[: max(len(order) // 2, min(20, len(order)))]      # burial-matched half
        for c in rng.choice(pool, size=min(n_ctrl, len(pool)), replace=False):
            dc = np.linalg.norm(st.ca - st.ca[c], axis=1)
            dc[np.isnan(dc)] = np.inf
            ctrl_sets.append(np.argsort(dc)[: len(main)])

    need = sorted({int(i) for i in main} | {int(i) for s in SHELLS for i in pocket[s]}
                  | {int(i) for cs in ctrl_sets for i in cs})
    sa = sasa(st, n_sphere=n_sphere, res_idx=need)

    row = {"accession": acc, "status": "ok", "anchor": anchor, "length_model": len(st),
           "aa_at_nuc": st.aa[ri], "plddt_mean_all": float(st.plddt.mean())}
    for s in SHELLS:
        row.update(set_descriptors(st, pocket[s], sa, nc, f"pk{int(s)}"))
    if ctrl_sets:
        per = [set_descriptors(st, cs, sa, nc, "ct") for cs in ctrl_sets]
        keys = per[0].keys()
        row.update({k: float(np.mean([p[k] for p in per])) for k in keys})
        row["ct_n_patches"] = len(ctrl_sets)
    try:
        row["cavity_open_vol"], row["cavity_enclosed_vol"] = cavity_volume(st, og, enclosure=enclosure)
    except Exception:  # noqa: BLE001
        row["cavity_open_vol"] = row["cavity_enclosed_vol"] = np.nan
    return row


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--structures-dir", default=None, help="structure directory relative to paths.root (default: paths.structures); point this at data/structures_esmfold for the ESMFold replication")
    ap.add_argument("--tag", default="", help='suffix added to output filenames (e.g. _esmfold) so runs do not overwrite each other')
    ap.add_argument("--radius", type=float, default=10.0, help="pocket shell used as the primary definition")
    ap.add_argument("--n-controls", type=int, default=8)
    ap.add_argument("--control-min-dist", type=float, default=15.0)
    ap.add_argument("--enclosure", type=int, default=55,
                    help="heavy atoms within 8 A for a cavity point to count as enclosed rather than bulk solvent; "
                         "calibrate on real models before trusting cavity_enclosed_vol")
    ap.add_argument("--n-sphere", type=int, default=512, help="SASA sphere points (512 gives about 1%% median error)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()
    cfg = load_config(args.config)

    tg = pd.read_csv(rpath(cfg, "processed", "structure_targets.tsv"), sep="\t")
    tg = tg.iloc[args.shard :: args.n_shards]
    sdir = (str(Path(cfg["paths"]["root"]).expanduser() / args.structures_dir)
            if args.structures_dir else str(Path(rpath(cfg, "structures", "x")).parent))
    print(f"structures: {sdir}")
    jobs = [(r.accession, r.nuc_pos, sdir, args.radius, args.n_controls, args.control_min_dist, args.n_sphere, args.enclosure)
            for r in tg.itertuples()]
    print(f"shard {args.shard}/{args.n_shards}: {len(jobs)} proteins, {args.workers} workers", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(one_protein, jobs, chunksize=8), 1):
            if r is not None:
                rows.append(r)
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)
    df = pd.DataFrame(rows)
    df["n_shards"] = args.n_shards          # lets stage 12 detect a truncated set of shard files
    suffix = "" if args.n_shards == 1 else f".{args.shard}"
    out = rpath(cfg, "processed", f"pocket_descriptors{args.tag}{suffix}.tsv")
    df.to_csv(out, sep="\t", index=False)
    print(f"\nwrote {len(df)} rows -> {out}")
    if "status" in df:
        print(df["status"].value_counts().to_string())
    ok = df[df.get("status", "ok") == "ok"]
    if len(ok):
        print(f"\ncatalytic residue really is Ser in the model: {(ok['aa_at_nuc'] == 'S').mean():.1%}")
        print(f"missing structures (no model on disk): {len(tg) - len(df)}")
        cols = [c for c in ("pk10_n", "pk10_hydropathy", "pk10_net_charge", "pk10_sasa_total", "cavity_open_vol", "cavity_enclosed_vol",
                            "ct_n", "ct_hydropathy", "ct_net_charge", "pk10_plddt") if c in ok]
        print("\n" + ok[cols].describe().round(2).to_string())


if __name__ == "__main__":
    main()
