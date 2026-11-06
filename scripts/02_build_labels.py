#!/usr/bin/env python
"""Attach temperature and specificity labels to the sequence table.

Inputs (all optional except sequences.tsv; missing tables just give unlabeled columns):
  data/raw/ogt_table.csv   columns: taxid (optional), organism, ogt
  data/raw/tm_table.csv    columns: accession, tm
  data/raw/cleavages.tsv   columns: protease_acc, window   (8-mer P4 P3 P2 P1 P1' P2' P3' P4')

Outputs:
  data/processed/labels.tsv     one row per accession
  data/processed/profiles.npz   accs, counts [N,P,20], n  (only proteases with observed cleavages)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serprot.common import (  # noqa: E402
    base_parser, cfg_file, counts_to_profile, load_config, norm_species, profile_breadth,
    profile_p1_breadth, rpath, window_counts,
)

ALIASES = {
    "taxid": ["taxid", "tax_id", "taxonomy_id", "ncbi_taxid", "species_taxid"],
    "organism": ["organism", "species", "species_name", "name", "organism_name"],
    "ogt": ["ogt", "growth_temp", "growth_temperature", "topt", "optimal_growth_temperature", "temperature"],
}


def pick(df: pd.DataFrame, key: str):
    low = {c.lower().strip(): c for c in df.columns}
    for a in ALIASES[key]:
        if a in low:
            return low[a]
    return None


def attach_ogt(seqs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    seqs = seqs.copy()
    seqs["ogt"], seqs["ogt_match"] = np.nan, "none"
    path = cfg_file(cfg, cfg["ogt"]["table"])
    if not path.exists():
        print(f"WARNING: OGT table not found at {path}; no temperature labels will be attached.")
        return seqs
    t = pd.read_csv(path)
    c_tax, c_org, c_ogt = pick(t, "taxid"), pick(t, "organism"), pick(t, "ogt")
    if c_ogt is None or (c_tax is None and c_org is None):
        raise SystemExit(f"{path}: need an OGT column and a taxid or organism column; got {list(t.columns)}")
    t = t[(t[c_ogt] >= cfg["ogt"]["min_ogt"]) & (t[c_ogt] <= cfg["ogt"]["max_ogt"])]
    if c_tax:
        by_tax = t.dropna(subset=[c_tax]).assign(_k=lambda d: d[c_tax].astype(float).astype(int)).groupby("_k")[c_ogt].median()
        k = pd.to_numeric(seqs["taxid"], errors="coerce")
        hit = k.map(by_tax)
        seqs.loc[hit.notna(), "ogt"] = hit[hit.notna()]
        seqs.loc[hit.notna(), "ogt_match"] = "taxid"
    if c_org:
        by_sp = t.assign(_k=t[c_org].map(norm_species)).query("_k != ''").groupby("_k")[c_ogt].agg(["median", "std", "count"])
        # skip species whose entries disagree a lot (>10 C): likely strain-dependent or mislabeled
        by_sp = by_sp[(by_sp["std"].fillna(0) <= 10)]["median"]
        need = seqs["ogt"].isna()
        hit = seqs.loc[need, "organism"].map(norm_species).map(by_sp)
        seqs.loc[hit.index[hit.notna()], "ogt"] = hit[hit.notna()]
        seqs.loc[hit.index[hit.notna()], "ogt_match"] = "species"
    print(seqs["ogt_match"].value_counts().to_string())
    return seqs


def attach_tm(seqs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    seqs = seqs.copy()
    seqs["tm"] = np.nan
    path = cfg_file(cfg, cfg["tm"]["table"])
    if path.exists():
        t = pd.read_csv(path)
        t.columns = [c.lower().strip() for c in t.columns]
        m = t.groupby("accession")["tm"].median()
        seqs["tm"] = seqs["accession"].map(m)
        print(f"Tm labels: {seqs['tm'].notna().sum()}")
    return seqs


def build_profiles(seqs: pd.DataFrame, cfg: dict):
    sc = cfg["specificity"]
    pos = sc["positions"]
    path = cfg_file(cfg, sc["cleavages"])
    seqs = seqs.copy()
    for c in ["spec_n", "breadth", "p1_breadth"]:
        seqs[c] = np.nan
    if not path.exists():
        print(f"WARNING: cleavage table not found at {path}; no specificity labels.")
        return seqs, None
    cl = pd.read_csv(path, sep="\t", dtype=str)
    cl.columns = [c.lower().strip() for c in cl.columns]
    if not {"protease_acc", "window"} <= set(cl.columns):
        raise SystemExit(f"{path} needs columns protease_acc and window; got {list(cl.columns)}")
    known = set(seqs["accession"])
    cl = cl[cl["protease_acc"].isin(known)]
    print(f"Cleavage windows mapped to our proteases: {len(cl)} over {cl['protease_acc'].nunique()} proteases")

    accs, counts, ns = [], [], []
    for acc, g in cl.groupby("protease_acc"):
        c = window_counts(g["window"].tolist(), pos)
        n = int(c[0].sum()) if len(pos) else 0
        if n >= sc["min_cleavages"]:
            accs.append(acc)
            counts.append(c)
            ns.append(n)
    print(f"Proteases with >= {sc['min_cleavages']} cleavages: {len(accs)}")
    if not accs:
        return seqs, None
    counts = np.stack(counts)
    probs = counts_to_profile(counts, sc["pseudocount"])
    p1 = pos.index(3) if 3 in pos else 0
    lab = pd.DataFrame({"accession": accs, "spec_n": ns, "breadth": profile_breadth(probs),
                        "p1_breadth": profile_p1_breadth(probs, p1)}).set_index("accession")
    for c in ["spec_n", "breadth", "p1_breadth"]:
        seqs[c] = seqs["accession"].map(lab[c])
    return seqs, {"accs": np.array(accs), "counts": counts, "n": np.array(ns)}


def main():
    ap = base_parser(__doc__)
    cfg = load_config(ap.parse_args().config)
    seqs = pd.read_csv(rpath(cfg, "processed", "sequences.tsv"), sep="\t")
    seqs = attach_ogt(seqs, cfg)
    seqs = attach_tm(seqs, cfg)
    seqs, prof = build_profiles(seqs, cfg)
    out = rpath(cfg, "processed", "labels.tsv")
    seqs.to_csv(out, sep="\t", index=False)
    if prof is not None:
        np.savez(rpath(cfg, "processed", "profiles.npz"), **prof)
    summ = seqs.groupby("family").agg(n=("accession", "size"), ogt=("ogt", "count"), tm=("tm", "count"), spec=("spec_n", "count"))
    print(summ.to_string())
    print(f"Wrote {out}")
    if summ["spec"].sum() < 100:
        print("NOTE: fewer than 100 proteases carry specificity labels; the specificity head will be weak. "
              "See README 'Data caveats'.")


if __name__ == "__main__":
    main()
