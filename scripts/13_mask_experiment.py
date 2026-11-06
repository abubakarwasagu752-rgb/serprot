#!/usr/bin/env python
"""Does the ESM-2 temperature signal live OUTSIDE the substrate-binding pocket? A sequence-level test.

The structural analysis says pocket composition barely tracks growth temperature while the rest of the
protein does. If that is real, then hiding the pocket from a sequence model should cost little temperature
information, while hiding an equally large patch elsewhere should cost a lot.

Conditions (all masked sets are the SAME SIZE for a given protein, so the comparison is fair):
  full          unmasked sequence (reuses the embedding cache from stage 4)
  pocket        pocket residues replaced by <mask>
  control_k     burial-matched patch far from the active site, replaced by <mask>
  random        a random set of residues, replaced by <mask>  (floor: what masking anything costs)
Each condition is embedded, a ridge model predicts OGT on the SAME folds, and the drop in Pearson r is
compared. Clade bootstrap gives a CI on the difference of drops.

  python scripts/13_mask_experiment.py --config config.yaml --n-controls 2
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serprot.common import base_parser, load_config, load_dataset, prepare_inputs, rpath, seed_all  # noqa: E402
from serprot.pockets import regions_for, to_sequence_positions  # noqa: E402
from serprot.structure import read_pdb  # noqa: E402


def build_masks(cfg, df, sdir, radius, n_controls, rng):
    """-> DataFrame with one row per usable protein and the 0-based positions to mask per condition."""
    rows = []
    miss = {"no_structure": 0, "no_regions": 0, "empty_after_crop": 0}
    for i, r in enumerate(df.itertuples()):
        f = sdir / f"{r.accession}.pdb.gz"
        if not f.exists():
            miss["no_structure"] += 1
            continue
        try:
            st = read_pdb(f)
        except Exception:  # noqa: BLE001
            miss["no_structure"] += 1
            continue
        reg = regions_for(st, r.nuc_pos, r.accession, radius=radius, n_controls=max(n_controls, 1))
        if reg is None or not reg["controls"]:
            miss["no_regions"] += 1
            continue
        L = len(r.seq_in)
        pk = to_sequence_positions(st, reg["pocket"], r.crop_off, L)
        cts = [to_sequence_positions(st, c, r.crop_off, L) for c in reg["controls"][:n_controls]]
        if len(pk) < 5 or any(len(c) < 5 for c in cts):
            miss["empty_after_crop"] += 1
            continue
        # every masked set gets the same size as the pocket, so no condition is handicapped
        k = min([len(pk)] + [len(c) for c in cts])
        rows.append({"row": i, "accession": r.accession, "pocket": pk[:k],
                     **{f"control_{j}": c[:k] for j, c in enumerate(cts)},
                     "random": rng.choice(L, size=min(k, L), replace=False), "n_masked": k})
        if (len(rows) + sum(miss.values())) % 2000 == 0:
            print(f"  prepared {len(rows)} / scanned {len(rows) + sum(miss.values())}", flush=True)
    print(f"maskable proteins: {len(rows)}   skipped: {miss}")
    return pd.DataFrame(rows)


def embed_masked(seqs, mask_sets, cfg, cache: Path, batch=16):
    """Mean-pooled ESM-2 embeddings with the given positions replaced by <mask>."""
    if cache.exists():
        E = np.load(cache)
        if len(E) == len(seqs):
            print(f"  reusing {cache.name}")
            return E
    import torch
    from transformers import EsmModel, EsmTokenizer

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = EsmTokenizer.from_pretrained(cfg["esm_model"])
    model = EsmModel.from_pretrained(cfg["esm_model"], add_pooling_layer=False).to(dev).eval()
    out = np.zeros((len(seqs), model.config.hidden_size), dtype=np.float32)
    order = np.argsort([len(s) for s in seqs])
    t0 = time.time()
    with torch.no_grad():
        for bi in range(0, len(order), batch):
            idx = order[bi : bi + batch]
            enc = tok([seqs[j] for j in idx], padding=True, return_tensors="pt").to(dev)
            ids = enc["input_ids"]
            for row, j in enumerate(idx):
                p = np.asarray(mask_sets[j], dtype=int)
                p = p[p < ids.shape[1] - 2]
                ids[row, p + 1] = tok.mask_token_id          # +1 skips <cls>
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                h = model(input_ids=ids, attention_mask=enc["attention_mask"]).last_hidden_state.float()
            m = enc["attention_mask"].clone()
            m[:, 0] = 0
            m[torch.arange(m.size(0), device=dev), enc["attention_mask"].sum(1) - 1] = 0
            m = m.unsqueeze(-1).float()
            out[idx] = ((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy()
            if (bi // batch) % 100 == 0:
                done = bi + len(idx)
                print(f"    {done}/{len(seqs)}  {done / max(time.time() - t0, 1):.0f}/s", flush=True)
    np.save(cache, out)
    return out


def cv_predict(X, y, folds):
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    pred = np.full(len(y), np.nan)
    for k in np.unique(folds):
        te, tr = folds == k, (folds != k) & np.isfinite(y)
        if tr.sum() < 20 or te.sum() == 0:
            continue
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(sc.transform(X[tr]), y[tr])
        pred[te] = m.predict(sc.transform(X[te]))
    return pred


def pearson(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 5 else np.nan


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--radius", type=float, default=10.0)
    ap.add_argument("--n-controls", type=int, default=2)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_all(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    full = prepare_inputs(load_dataset(cfg), cfg["max_len"])
    sdir = Path(rpath(cfg, "structures", "x")).parent
    tg = set(pd.read_csv(rpath(cfg, "processed", "structure_targets.tsv"), sep="\t")["accession"])
    df = full[full["accession"].isin(tg) & full["nuc_pos"].notna() & full["ogt"].notna()].reset_index(drop=True)
    print(f"{len(df)} candidate proteins (structure target, catalytic Ser, OGT label)")

    masks = build_masks(cfg, df, sdir, args.radius, args.n_controls, rng)
    if masks.empty:
        raise SystemExit("no proteins could be masked")
    sub = df.iloc[masks["row"].values].reset_index(drop=True)
    seqs = sub["seq_in"].tolist()
    y, folds = sub["ogt"].values.astype(float), sub["fold_cv"].values
    print(f"masking {masks['n_masked'].median():.0f} residues per protein (median), "
          f"{100 * masks['n_masked'].median() / np.median([len(s) for s in seqs]):.1f}% of sequence")

    conditions = {"pocket": "pocket", "random": "random",
                  **{f"control_{j}": f"control_{j}" for j in range(args.n_controls)}}
    # the unmasked baseline comes from stage 4's cache when available
    import re

    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(cfg["esm_model"]).rstrip("/").split("/")[-1]).strip("_")
    cache_full = Path(rpath(cfg, "processed", f"esm_embeddings_{slug}.npy"))
    X = {}
    if cache_full.exists():
        E_all = np.load(cache_full)
        if len(E_all) == len(full):
            pos = pd.Series(np.arange(len(full)), index=full["accession"].values)
            X["full"] = E_all[pos.loc[sub["accession"].values].to_numpy()]
            print(f"  reusing unmasked embeddings from stage 4 ({cache_full.name})")
        else:
            print(f"  stage-4 cache has {len(E_all)} rows but dataset has {len(full)}; re-embedding")
    if "full" not in X:
        X["full"] = embed_masked(seqs, [np.array([], dtype=int)] * len(seqs), cfg,
                                 rpath(cfg, "processed", f"maskemb_full_{slug}.npy"), args.batch)
    for name, col in conditions.items():
        print(f"embedding condition: {name}")
        X[name] = embed_masked(seqs, masks[col].tolist(), cfg,
                               rpath(cfg, "processed", f"maskemb_{name}_r{int(args.radius)}_{slug}.npy"),
                               args.batch)

    preds = {k: cv_predict(v, y, folds) for k, v in X.items()}
    r = {k: pearson(p, y) for k, p in preds.items()}
    ctrl_keys = [k for k in r if k.startswith("control_")]
    print("\n==== temperature prediction (Pearson r, held-out folds) ====")
    for k in ["full", "pocket", *ctrl_keys, "random"]:
        drop = r["full"] - r[k]
        print(f"  {k:>12}: r = {r[k]:.4f}   drop vs full = {drop:+.4f}")

    # clade bootstrap on the difference of drops
    clades = sub["cluster_pairs"].values
    uc = np.unique(clades)
    members = {c: np.flatnonzero(clades == c) for c in uc}
    diffs = []
    for _ in range(args.n_boot):
        pick = rng.choice(uc, size=len(uc), replace=True)
        idx = np.concatenate([members[c] for c in pick])
        rp = pearson(preds["pocket"][idx], y[idx])
        rc = np.mean([pearson(preds[k][idx], y[idx]) for k in ctrl_keys])
        rf = pearson(preds["full"][idx], y[idx])
        diffs.append((rf - rc) - (rf - rp))          # control drop minus pocket drop
    diffs = np.array([d for d in diffs if np.isfinite(d)])
    obs = (r["full"] - np.mean([r[k] for k in ctrl_keys])) - (r["full"] - r["pocket"])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"\ncontrol drop minus pocket drop = {obs:+.4f}  (95% CI {lo:+.4f} to {hi:+.4f})")
    print("  positive and excluding 0 => masking the pocket costs LESS temperature information")
    print("  than masking an equal, burial-matched patch elsewhere: the thermal signal avoids the pocket.")

    out = rpath(cfg, "runs", "analysis", f"mask_experiment_r{int(args.radius)}.tsv")
    pd.DataFrame([{"condition": k, "pearson_r": r[k], "drop_vs_full": r["full"] - r[k],
                   "n": int(np.isfinite(preds[k]).sum())} for k in r]).to_csv(out, sep="\t", index=False)
    pd.DataFrame({"boot_diff": diffs}).to_csv(str(out).replace(".tsv", "_boot.tsv"), sep="\t", index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
