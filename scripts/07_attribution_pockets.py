#!/usr/bin/env python
"""Where do the thermal and specificity heads look, and do they collide in the substrate-binding region?

For held-out proteins of each fold (so the model never saw them):
  1. per-residue attribution = || d(output)/d(embedding) * embedding ||  for the OGT head and the breadth head
  2. pocket = residues within pocket.shell_angstrom of the catalytic Ser OG in the AlphaFold model
     (or user-supplied via --pocket-file: accession <TAB> comma-separated 1-based positions)
  3. enrichment tests against size-matched control sets (spatial shells around random residues when coordinates are
     available, random residue sets otherwise):
        pocket_percentile   fraction of control sets with less attribution mass than the pocket
        overlap             top-X% residues of BOTH heads vs the chance expectation k^2/L
Outputs: runs/analysis/attribution_per_protein.tsv, attribution_stats.tsv, attributions.npz, figures/pocket_enrichment.png

Caveat: saliency is a correlational probe of the trained model, and the 8 A shell is a coarse stand-in for S1-S4.
Substrate-complex structures (PDB) give better pockets; pass them with --pocket-file.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serprot.common import base_parser, load_config, load_dataset, prepare_inputs, rpath, seed_all  # noqa: E402
from serprot.model import breadth_from_logits, load_trained  # noqa: E402


# ------------------------------------------------------------------ structures
def fetch_alphafold(acc: str, sdir: Path) -> Path | None:
    f = sdir / f"{acc}.pdb"
    if f.exists():
        return f
    try:
        r = requests.get(f"https://alphafold.ebi.ac.uk/api/prediction/{acc}", timeout=30)
        r.raise_for_status()
        url = r.json()[0]["pdbUrl"]
        f.write_bytes(requests.get(url, timeout=60).content)
        return f
    except Exception:  # noqa: BLE001
        return None


def read_structure(path: Path):
    from Bio.PDB import PDBParser

    chain = next(PDBParser(QUIET=True).get_structure("x", str(path))[0].get_chains())
    pos, atoms, ca, plddt, og = [], [], [], [], {}
    for res in chain:
        if res.id[0] != " " or "CA" not in res:
            continue
        pos.append(res.id[1])
        atoms.append(np.array([a.coord for a in res]))
        ca.append(res["CA"].coord)
        plddt.append(res["CA"].bfactor)
        if "OG" in res:
            og[res.id[1]] = res["OG"].coord
    return np.array(pos), atoms, np.array(ca), np.array(plddt), og


def pocket_from_structure(st, nuc_pos, shell, min_plddt):
    pos, atoms, ca, plddt, og = st
    if nuc_pos not in og:
        return None
    d = np.array([np.min(np.linalg.norm(a - og[nuc_pos], axis=1)) for a in atoms])
    sel = (d <= shell) & (plddt >= min_plddt)
    return set(pos[sel].tolist())


# ------------------------------------------------------------------ saliency
def saliency(model, tok, seq, dev):
    enc = tok(seq, return_tensors="pt").to(dev)
    with torch.enable_grad():
        out = model(enc["input_ids"], enc["attention_mask"])
        emb = model._emb_out
        res = {}
        for name, scalar in [("T", out["ogt"].sum()), ("S", breadth_from_logits(out["spec"]).sum())]:
            g = torch.autograd.grad(scalar, emb, retain_graph=True)[0]
            s = (g * emb).norm(dim=-1)[0, 1:-1].detach().cpu().numpy()  # drop <cls>/<eos>
            res[name] = s / max(s.sum(), 1e-12)
    return res


def mass_percentile(s, pocket_idx, controls):
    obs = s[pocket_idx].sum()
    cm = np.array([s[c].sum() for c in controls])
    return float((cm < obs).mean() + 0.5 * (cm == obs).mean()), float(obs)


def make_controls(L, pocket_idx, ca_window, rng, n=50):
    k = len(pocket_idx)
    inside = set(pocket_idx.tolist())
    outside = np.array([i for i in range(L) if i not in inside])
    controls = []
    if ca_window is not None and len(outside) > 0:
        for c in rng.choice(outside, size=min(n, len(outside)), replace=False):
            d = np.linalg.norm(ca_window - ca_window[c], axis=1)
            controls.append(np.argsort(d)[:k])
    else:
        for _ in range(n):
            controls.append(rng.choice(L, size=k, replace=False))
    return controls


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--variant", default="disent")
    ap.add_argument("--split", default="cv")
    ap.add_argument("--pocket-file", default=None)
    ap.add_argument("--no-download", action="store_true", help="never contact AlphaFold DB (needs --pocket-file)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_all(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import EsmTokenizer

    tok = EsmTokenizer.from_pretrained(cfg["esm_model"])
    df = prepare_inputs(load_dataset(cfg), cfg["max_len"])
    df = df[df["nuc_pos"].notna()]
    ac = cfg["attribution"]
    pc = cfg["pocket"]
    sdir = rpath(cfg, "structures", "x").parent
    given = {}
    if args.pocket_file:
        pf = pd.read_csv(args.pocket_file, sep="\t")
        given = {a: set(int(x) for x in str(p).split(",") if x) for a, p in zip(pf["accession"], pf["positions"])}

    run_dir = rpath(cfg, "runs", f"{args.variant}_{args.split}")
    folds = sorted(run_dir.glob("fold*"))
    per_fold = int(np.ceil(ac["max_proteins_per_family"] / max(len(folds), 1)))
    rows, arrays = [], {}
    for fdir in folds:
        model, ck = load_trained(fdir / "model.pt", dev)
        hook = model.enable_embedding_grad()
        held = df[df["accession"].isin(ck["accessions"])]
        for fam, g in held.groupby("family"):
            g = g.sample(n=min(per_fold, len(g)), random_state=cfg["seed"])
            for _, r in g.iterrows():
                acc, off, nuc = r["accession"], int(r["crop_off"]), int(r["nuc_pos"])
                st, ca_window = None, None
                if acc in given:
                    pocket = given[acc]
                else:
                    if args.no_download:
                        continue
                    f = fetch_alphafold(acc, sdir)
                    if f is None:
                        continue
                    st = read_structure(f)
                    pocket = pocket_from_structure(st, nuc, pc["shell_angstrom"], pc["min_plddt"])
                    if pocket is None:
                        continue
                L = len(r["seq_in"])
                pidx = np.array(sorted(p - 1 - off for p in pocket if 0 <= p - 1 - off < L))
                if len(pidx) < 5 or len(pidx) > 0.6 * L:
                    continue
                if st is not None:
                    pos, _, ca, _, _ = st
                    lut = {p: i for i, p in enumerate(pos.tolist())}
                    idx = [lut.get(off + j + 1) for j in range(L)]
                    if all(i is not None for i in idx):
                        ca_window = ca[idx]
                s = saliency(model, tok, r["seq_in"], dev)
                controls = make_controls(L, pidx, ca_window, rng)
                row = dict(accession=acc, family=fam, fold=fdir.name, L=L, pocket_n=len(pidx), nuc_pos=nuc)
                for h in ("T", "S"):
                    pct, obs = mass_percentile(s[h], pidx, controls)
                    row[f"pocket_mass_{h}"], row[f"pocket_pct_{h}"] = obs, pct
                    row[f"log2_enrich_{h}"] = float(np.log2((obs + 1e-6) / (len(pidx) / L + 1e-6)))
                k = max(1, int(np.ceil(ac["top_fraction"] * L)))
                topT, topS = set(np.argsort(-s["T"])[:k].tolist()), set(np.argsort(-s["S"])[:k].tolist())
                inter = topT & topS
                row["overlap_n"], row["overlap_expected"] = len(inter), k * k / L
                row["log2_overlap_ratio"] = float(np.log2((len(inter) + 0.5) / (k * k / L + 0.5)))
                row["overlap_in_pocket_frac"] = float(np.mean([i in set(pidx.tolist()) for i in inter])) if inter else np.nan
                row["pocket_frac"] = len(pidx) / L
                rows.append(row)
                arrays[f"{acc}|T"], arrays[f"{acc}|S"], arrays[f"{acc}|pocket"] = s["T"], s["S"], pidx
        hook.remove()
        del model
        print(f"{fdir.name}: {len(rows)} proteins so far", flush=True)

    out = rpath(cfg, "runs", "analysis", "x").parent
    res = pd.DataFrame(rows)
    if res.empty:
        raise SystemExit("no proteins processed (structures unavailable? try --pocket-file)")
    res.to_csv(out / "attribution_per_protein.tsv", sep="\t", index=False)
    np.savez_compressed(out / "attributions.npz", **arrays)

    def w(x):  # Wilcoxon signed-rank vs 0 with guards
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if len(x) < 6 or np.allclose(x, 0):
            return np.nan
        return float(stats.wilcoxon(x).pvalue)

    srows = []
    for fam in ["all"] + sorted(res["family"].unique()):
        q = res if fam == "all" else res[res["family"] == fam]
        for h, name in (("T", "thermal head (OGT)"), ("S", "specificity head (breadth)")):
            srows.append(dict(family=fam, test=f"pocket percentile vs 0.5, {name}", n=len(q),
                              median=float(np.median(q[f"pocket_pct_{h}"])), p=w(q[f"pocket_pct_{h}"] - 0.5)))
        srows.append(dict(family=fam, test="thermal minus specificity pocket percentile", n=len(q),
                          median=float(np.median(q["pocket_pct_T"] - q["pocket_pct_S"])), p=w(q["pocket_pct_T"] - q["pocket_pct_S"])))
        srows.append(dict(family=fam, test="log2 overlap ratio of top residues vs chance", n=len(q),
                          median=float(np.median(q["log2_overlap_ratio"])), p=w(q["log2_overlap_ratio"])))
        srows.append(dict(family=fam, test="overlap residues in pocket minus pocket fraction", n=int(q["overlap_in_pocket_frac"].notna().sum()),
                          median=float(np.nanmedian(q["overlap_in_pocket_frac"] - q["pocket_frac"])),
                          p=w(q["overlap_in_pocket_frac"] - q["pocket_frac"])))
    st = pd.DataFrame(srows)
    st.to_csv(out / "attribution_stats.tsv", sep="\t", index=False)
    print(st.round(4).to_string(index=False))
    plot(res, rpath(cfg, "figures", "pocket_enrichment.png"))


def plot(res, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e3e2de", "#fcfcfb"
    heads = [("T", "Thermal head", "#2a78d6"), ("S", "Specificity head", "#eb6834")]
    fams = sorted(res["family"].unique())
    fig, ax = plt.subplots(figsize=(7, 0.9 * len(fams) + 1.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    rng = np.random.default_rng(0)
    for i, fam in enumerate(fams):
        q = res[res["family"] == fam]
        for j, (h, lab, c) in enumerate(heads):
            y = i + (j - 0.5) * 0.34
            v = q[f"pocket_pct_{h}"].values
            ax.scatter(v, y + rng.normal(0, 0.03, len(v)), s=10, color=c, alpha=0.35, edgecolor="none")
            ax.vlines(np.median(v), y - 0.14, y + 0.14, color=c, lw=2.5)
            ax.text(1.02, y, f"{np.median(v):.2f}", va="center", fontsize=8.5, color=INK, transform=ax.get_yaxis_transform())
    ax.axvline(0.5, color=INK2, lw=0.8, ls=":")
    ax.set_yticks(range(len(fams)))
    ax.set_yticklabels(fams, fontsize=9.5, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Pocket attribution percentile vs size-matched control regions (0.5 = chance)", fontsize=9, color=INK2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, lw=0.6)
    for j, (h, lab, c) in enumerate(heads):
        ax.scatter([], [], color=c, s=25, label=lab)
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)


if __name__ == "__main__":
    main()
