#!/usr/bin/env python
"""Train the dual-subspace ESM-2 model on one fold and predict the held-out proteins.

  python scripts/05_train_disentangled.py --variant disent --split cv --fold 0
  python scripts/05_train_disentangled.py --variant joint  --split lofo --fold 1   # hold out family index 1

Outputs in runs/<variant>_<split>/fold<k>/ : model.pt, preds.tsv, spec_probs.npy, zT.npy, zS.npy, log.json
Held-out rows = ALL sequences of the test fold (labeled or not), so downstream analyses can use them.
Training rows = labeled sequences (OGT, Tm or specificity) outside the test fold.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serprot.common import base_parser, load_config, load_dataset, load_profiles, prepare_inputs, rpath, seed_all  # noqa: E402
from serprot.model import VARIANTS, DisentangledESM, distance_correlation_sq  # noqa: E402


class SeqDS(Dataset):
    def __init__(self, df, profiles, n_pos, stats, phy_map):
        self.df = df.reset_index(drop=True)
        self.profiles, self.n_pos, self.stats, self.phy_map = profiles, n_pos, stats, phy_map
        self.cap = stats["weight_cap"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        prof = self.profiles.get(r["accession"])
        has_spec = prof is not None
        n = r["spec_n"] if has_spec and pd.notna(r["spec_n"]) else 0
        return {
            "seq": r["seq_in"],
            "ogt": (r["ogt"] - self.stats["ogt_mu"]) / self.stats["ogt_sd"] if pd.notna(r["ogt"]) else 0.0,
            "m_ogt": float(pd.notna(r["ogt"])),
            "w_ogt": float(r["w_ogt"]) if pd.notna(r["ogt"]) else 0.0,
            "tm": (r["tm"] - self.stats["tm_mu"]) / self.stats["tm_sd"] if pd.notna(r["tm"]) else 0.0,
            "m_tm": float(pd.notna(r["tm"])),
            "spec": prof if has_spec else np.full((self.n_pos, 20), 1 / 20),
            "m_spec": float(has_spec),
            "w_spec": min(n, self.cap) / self.cap if has_spec else 0.0,
            "phy": self.phy_map.get(r["phylum"], -1),
        }


def make_collate(tok, max_len):
    def collate(batch):
        enc = tok([b["seq"] for b in batch], padding=True, truncation=True, max_length=max_len + 2, return_tensors="pt")
        t = lambda k, dt=torch.float32: torch.tensor(np.array([b[k] for b in batch]), dtype=dt)  # noqa: E731
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "ogt": t("ogt"), "m_ogt": t("m_ogt"), "w_ogt": t("w_ogt"), "tm": t("tm"), "m_tm": t("m_tm"),
                "spec": t("spec"), "m_spec": t("m_spec"), "w_spec": t("w_spec"), "phy": t("phy", torch.long)}
    return collate


def wmean(x, w):
    return (x * w).sum() / w.sum().clamp(min=1e-6)


def soft_ce(logits, target):
    return -(target * F.log_softmax(logits.float(), -1)).sum(-1).mean(-1)


def compute_loss(out, b, cfg_t, variant, dev):
    ogt, tm = b["ogt"].to(dev), b["tm"].to(dev)
    m_ogt, m_tm, m_spec = b["m_ogt"].to(dev), b["m_tm"].to(dev), b["m_spec"].to(dev)
    w_ogt, w_spec, spec = b["w_ogt"].to(dev), b["w_spec"].to(dev), b["spec"].to(dev)
    parts = {}
    parts["ogt"] = wmean((out["ogt"].float() - ogt) ** 2, w_ogt * m_ogt) if m_ogt.sum() > 0 else out["ogt"].sum() * 0
    parts["tm"] = wmean((out["tm"].float() - tm) ** 2, m_tm) if m_tm.sum() > 0 else out["tm"].sum() * 0
    parts["spec"] = wmean(soft_ce(out["spec"], spec), w_spec * m_spec) if m_spec.sum() > 0 else out["spec"].sum() * 0
    loss = parts["ogt"] + cfg_t["tm_weight"] * parts["tm"] + cfg_t["spec_weight"] * parts["spec"]
    if "adv_ogt" in out:
        a1 = wmean((out["adv_ogt"].float() - ogt) ** 2, m_ogt) if m_ogt.sum() > 0 else out["adv_ogt"].sum() * 0
        a2 = wmean(soft_ce(out["adv_spec"], spec), w_spec * m_spec) if m_spec.sum() > 0 else out["adv_spec"].sum() * 0
        parts["adv_ogt_from_zS"], parts["adv_spec_from_zT"] = a1, a2
        loss = loss + a1 + a2
        if "adv_phy_T" in out and cfg_t["lam_phylum"] > 0:
            phy = b["phy"].to(dev)
            ok = (phy >= 0).float()
            if ok.sum() > 0:
                ce = lambda lg: wmean(F.cross_entropy(lg.float(), phy.clamp(min=0), reduction="none"), ok)  # noqa: E731
                parts["adv_phy"] = ce(out["adv_phy_T"]) + ce(out["adv_phy_S"])
                loss = loss + cfg_t["lam_phylum"] * parts["adv_phy"]
    if variant in ("disent", "noadv") and out["zT"].size(0) >= 8:
        parts["dcor"] = distance_correlation_sq(out["zT"], out["zS"])
        loss = loss + cfg_t["lam_dcor"] * parts["dcor"]
    return loss, parts


@torch.no_grad()
def predict(model, loader, dev, stats, use_amp):
    model.eval()
    res = {k: [] for k in ["ogt", "tm", "spec", "zT", "zS"]}
    for b in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            out = model(b["input_ids"].to(dev), b["attention_mask"].to(dev))
        res["ogt"].append((out["ogt"].float().cpu().numpy() * stats["ogt_sd"]) + stats["ogt_mu"])
        res["tm"].append((out["tm"].float().cpu().numpy() * stats["tm_sd"]) + stats["tm_mu"])
        res["spec"].append(F.softmax(out["spec"].float(), -1).cpu().numpy())
        res["zT"].append(out["zT"].float().cpu().numpy())
        res["zS"].append(out["zS"].float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in res.items()}


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--variant", default="disent", choices=VARIANTS)
    ap.add_argument("--split", default="cv", choices=["cv", "lofo"])
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ct = dict(cfg["train"])
    if args.epochs:
        ct["epochs"] = args.epochs
    seed_all(cfg["seed"] + args.fold)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = dev.type == "cuda"

    df = prepare_inputs(load_dataset(cfg), cfg["max_len"])
    profiles = load_profiles(cfg)
    pos = cfg["specificity"]["positions"]
    n_pos = len(pos)
    fcol = "fold_cv" if args.split == "cv" else "fold_lofo"
    test = df[df[fcol] == args.fold]
    labeled = df[(df[fcol] != args.fold) & (df["ogt"].notna() | df["tm"].notna() | df["accession"].isin(profiles))]
    # validation = random clusters held out of the training rows
    rng = np.random.default_rng(cfg["seed"] + args.fold)
    cl = labeled["cluster_split"].unique()
    val_cl = set(rng.choice(cl, size=max(1, int(len(cl) * ct["val_fraction"])), replace=False))
    val = labeled[labeled["cluster_split"].isin(val_cl)]
    train = labeled[~labeled["cluster_split"].isin(val_cl)].copy()
    if len(val) > ct.get("max_val", 10**9):
        val = val.sample(n=int(ct["max_val"]), random_state=cfg["seed"])
    print(f"[{args.variant} {args.split} fold {args.fold}] train={len(train)} val={len(val)} test={len(test)} device={dev}")

    stats = {"ogt_mu": float(train["ogt"].mean()) if train["ogt"].notna().any() else 0.0,
             "ogt_sd": float(train["ogt"].std()) if train["ogt"].notna().sum() > 1 else 1.0,
             "tm_mu": float(train["tm"].mean()) if train["tm"].notna().any() else 0.0,
             "tm_sd": float(train["tm"].std()) if train["tm"].notna().sum() > 1 else 1.0,
             "weight_cap": cfg["specificity"]["weight_cap"]}
    stats["ogt_sd"] = stats["ogt_sd"] or 1.0
    stats["tm_sd"] = stats["tm_sd"] or 1.0

    # OGT balancing weights (5 C bins), normalised to mean 1 over labeled rows
    for d in (train, val, test):
        d["w_ogt"] = 1.0
    if ct["balance_ogt"] and train["ogt"].notna().any():
        bins = (train["ogt"] // 5).astype("float")
        cnt = bins.map(bins.value_counts())
        w = 1 / np.sqrt(cnt)
        w = w / w[train["ogt"].notna()].mean()
        train["w_ogt"] = w.fillna(0).values
    phy_counts = train["phylum"].value_counts()
    phy_map = {p: i for i, p in enumerate(phy_counts[phy_counts >= 5].index)}
    n_phyla = len(phy_map) if ct["lam_phylum"] > 0 else 0

    from transformers import EsmTokenizer

    tok = EsmTokenizer.from_pretrained(cfg["esm_model"])
    coll = make_collate(tok, cfg["max_len"])
    mk = lambda d: SeqDS(d, profiles, n_pos, stats, phy_map)  # noqa: E731
    sw = np.where(train["accession"].isin(profiles), ct["spec_oversample"], 1.0)
    n_draw = len(train)
    if ct.get("steps_per_epoch"):
        n_draw = min(len(train), int(ct["steps_per_epoch"]) * ct["batch_size"])
    sampler = WeightedRandomSampler(sw, num_samples=n_draw, replacement=True)
    nw = ct["num_workers"]
    dl_train = DataLoader(mk(train), batch_size=ct["batch_size"], sampler=sampler, collate_fn=coll, num_workers=nw, drop_last=len(train) > ct["batch_size"])
    dl_val = DataLoader(mk(val), batch_size=ct["batch_size"] * 2, collate_fn=coll, num_workers=nw)
    dl_test = DataLoader(mk(test), batch_size=ct["batch_size"] * 2, collate_fn=coll, num_workers=nw)

    model = DisentangledESM(cfg["esm_model"], n_pos, n_phyla, args.variant, ct["latent_dim"], ct["hidden_dim"],
                            ct["dropout"], ct["unfreeze_last"]).to(dev)
    bb = [p for p in model.backbone.parameters() if p.requires_grad]
    hd = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": ct["lr_backbone"]}, {"params": hd, "lr": ct["lr_head"]}], weight_decay=ct["weight_decay"])

    best, best_state, bad, log = float("inf"), None, 0, []
    t_start, budget = time.time(), float(ct.get("time_budget_hours", 1e9)) * 3600
    last_ep_secs = 0.0
    for ep in range(ct["epochs"]):
        if best_state is not None and (time.time() - t_start) + 1.2 * last_ep_secs > budget:
            print(f"time budget of {budget / 3600:.1f} h reached; stopping before epoch {ep}", flush=True)
            break
        model.train()
        t0, agg = time.time(), {}
        for b in dl_train:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                out = model(b["input_ids"].to(dev), b["attention_mask"].to(dev), lam_adv=ct["lam_adv"])
            loss, parts = compute_loss(out, b, ct, args.variant, dev)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
            opt.step()
            for k, v in parts.items():
                agg.setdefault(k, []).append(float(v.detach()))
        # validation (task losses only, no adversaries)
        model.eval()
        vl, nv = 0.0, 0
        with torch.no_grad():
            for b in dl_val:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    out = model(b["input_ids"].to(dev), b["attention_mask"].to(dev))
                l, p = compute_loss(out, b, {**ct, "lam_phylum": 0}, "plain", dev)
                vl += float(p["ogt"] + ct["spec_weight"] * p["spec"]) * len(b["ogt"])
                nv += len(b["ogt"])
        vl /= max(nv, 1)
        row = {"epoch": ep, "val": vl, "secs": round(time.time() - t0, 1), **{k: float(np.mean(v)) for k, v in agg.items()}}
        log.append(row)
        last_ep_secs = time.time() - t0
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.trainable_state().items()}
        else:
            bad += 1
            if bad >= ct["patience"]:
                print("early stop")
                break

    model.load_state_dict(best_state, strict=False)
    res = predict(model, dl_test, dev, stats, use_amp)
    outdir = rpath(cfg, "runs", f"{args.variant}_{args.split}", f"fold{args.fold}", "x")  # 'x' makes the dir
    outdir = outdir.parent
    from serprot.common import profile_breadth

    pred = pd.DataFrame({"accession": test["accession"].values, "fold": args.fold, "pred_ogt": res["ogt"], "pred_tm": res["tm"],
                         "pred_breadth": profile_breadth(res["spec"])})
    pred.to_csv(outdir / "preds.tsv", sep="\t", index=False)
    np.save(outdir / "spec_probs.npy", res["spec"])
    np.save(outdir / "zT.npy", res["zT"])
    np.save(outdir / "zS.npy", res["zS"])
    torch.save({"state": best_state, "variant": args.variant, "n_pos": n_pos, "n_phyla": n_phyla, "stats": stats,
                "latent": ct["latent_dim"], "hidden": ct["hidden_dim"], "dropout": ct["dropout"],
                "unfreeze_last": ct["unfreeze_last"], "esm_model": cfg["esm_model"], "accessions": test["accession"].tolist()},
               outdir / "model.pt")
    json.dump(log, open(outdir / "log.json", "w"), indent=1)
    print(f"Saved -> {outdir}")


if __name__ == "__main__":
    main()
