#!/usr/bin/env python
"""Baselines that the temperature-aware dual-subspace model must beat.

  physchem : amino-acid composition + length + charged / IVYWREL fractions -> ridge
  esm      : frozen ESM-2 mean-pooled embeddings -> ridge          (needs torch; caches embeddings.npy)

Targets: OGT (scalar) and the specificity profile [P,20] (ridge on frequencies, then clipped + renormalised).
Evaluation uses the same lineage-aware folds as the neural model (cv or lofo).
Writes runs/baseline_<features>_<split>/preds.tsv with the same columns as the neural model.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serprot.common import (  # noqa: E402
    AA, base_parser, load_config, load_dataset, load_profiles, prepare_inputs, profile_breadth, rpath, safe_corr, seed_all,
)


def physchem(seqs):
    X = np.zeros((len(seqs), 23))
    for i, s in enumerate(seqs):
        n = len(s)
        X[i, :20] = [s.count(a) / n for a in AA]
        X[i, 20] = np.log(n)
        X[i, 21] = sum(s.count(a) for a in "DEKR") / n
        X[i, 22] = sum(s.count(a) for a in "IVYWREL") / n
    return X


def model_slug(cfg) -> str:
    """Cache file tag so embeddings from different ESM models are never mixed up."""
    import re

    return re.sub(r"[^A-Za-z0-9]+", "_", str(cfg["esm_model"]).rstrip("/").split("/")[-1]).strip("_")


def esm_embeddings(df, cfg, cache: Path):
    if cache.exists():
        E = np.load(cache)
        if len(E) == len(df):
            return E
    import torch
    from transformers import EsmModel, EsmTokenizer

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = EsmTokenizer.from_pretrained(cfg["esm_model"])
    model = EsmModel.from_pretrained(cfg["esm_model"], add_pooling_layer=False).to(dev).eval()
    seqs = df["seq_in"].tolist()
    order = np.argsort([len(s) for s in seqs])
    out = np.zeros((len(seqs), model.config.hidden_size), dtype=np.float32)
    bs = 32
    with torch.no_grad():
        for i in range(0, len(order), bs):
            idx = order[i : i + bs]
            enc = tok([seqs[j] for j in idx], padding=True, return_tensors="pt").to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                h = model(**enc).last_hidden_state.float()
            m = enc["attention_mask"].clone()
            m[:, 0] = 0
            m[torch.arange(m.size(0), device=dev), enc["attention_mask"].sum(1) - 1] = 0
            m = m.unsqueeze(-1).float()
            out[idx] = ((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy()
            if (i // bs) % 50 == 0:
                print(f"  embedded {i}/{len(order)}", flush=True)
    np.save(cache, out)
    return out


def run(df, X, profiles, positions, split, name, cfg):
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    fcol = "fold_cv" if split == "cv" else "fold_lofo"
    n_pos = len(positions)
    spec_acc = df["accession"].isin(profiles).values
    Y_spec = np.stack([profiles[a].reshape(-1) if a in profiles else np.full(n_pos * 20, 1 / 20) for a in df["accession"]])
    preds = []
    for k in sorted(df[fcol].unique()):
        te = (df[fcol] == k).values
        tr = ~te
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        po = np.full(te.sum(), np.nan)
        m = tr & df["ogt"].notna().values
        if m.sum() > 10:
            r = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(sc.transform(X[m]), df.loc[m, "ogt"].values)
            po = r.predict(Xte)
        ps = np.full((te.sum(), n_pos, 20), 1 / 20)
        m = tr & spec_acc
        if m.sum() > 10:
            r = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(sc.transform(X[m]), Y_spec[m])
            raw = np.clip(r.predict(Xte), 1e-4, None).reshape(-1, n_pos, 20)
            ps = raw / raw.sum(-1, keepdims=True)
        sub = pd.DataFrame({"accession": df.loc[te, "accession"].values, "fold": k, "pred_ogt": po,
                            "pred_tm": np.nan, "pred_breadth": profile_breadth(ps)})
        preds.append(sub)
    preds = pd.concat(preds)
    out = rpath(cfg, "runs", f"baseline_{name}_{split}", "preds.tsv")
    preds.to_csv(out, sep="\t", index=False)
    j = df[["accession", "ogt", "breadth"]].merge(preds, on="accession")
    print(f"[{name} {split}] OGT r={safe_corr(j['ogt'], j['pred_ogt']):.3f}  breadth r={safe_corr(j['breadth'], j['pred_breadth']):.3f} -> {out}")


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--features", nargs="+", default=["physchem", "esm"], choices=["physchem", "esm"])
    ap.add_argument("--splits", nargs="+", default=["cv", "lofo"])
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_all(cfg["seed"])
    df = prepare_inputs(load_dataset(cfg), cfg["max_len"])
    profiles = load_profiles(cfg)
    for feat in args.features:
        X = physchem(df["seq_in"].tolist()) if feat == "physchem" else esm_embeddings(df, cfg, rpath(cfg, "processed", f"esm_embeddings_{model_slug(cfg)}.npy"))
        for split in args.splits:
            run(df, X, profiles, cfg["specificity"]["positions"], split, feat, cfg)


if __name__ == "__main__":
    main()
