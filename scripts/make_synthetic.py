#!/usr/bin/env python
"""Generate a small synthetic dataset with a *planted* thermostability-specificity trade-off.

Purpose: test that every pipeline stage runs, and that the trade-off analysis recovers a known effect.
It writes the files that stages 01-02 would normally produce/consume, so the pipeline can start at stage 03.

Planted structure
  - clades of related sequences (mutated copies of a clade root)
  - OGT varies between and within clades
  - 'thermal' surface positions: charged-residue probability rises with OGT
  - 'pocket' positions: bulky-residue probability rises with OGT only partly (the trade-off, strength --tradeoff)
  - specificity breadth falls with pocket bulkiness; cleavage windows are sampled from the resulting profile
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

AA = "ACDEFGHIKLMNPQRSTVWY"
CHARGED, UNCHARGED = list("EKR"), list("STNQGA")
BULKY, SMALL = list("FWYLIM"), list("GASTV")
MOTIF = {"S8": "GTSMAAP", "S9": "GWSYGG", "S1": "GDSGGP"}
SER_OFF = {"S8": 2, "S9": 2, "S1": 1}
FAMS = list(MOTIF)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="project root for the synthetic run")
    ap.add_argument("--clades", type=int, default=45)
    ap.add_argument("--members", type=int, default=10)
    ap.add_argument("--tradeoff", type=float, default=0.6, help="0 = independent, 1 = strong coupling")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    root = Path(a.root)
    (root / "data/raw").mkdir(parents=True, exist_ok=True)
    (root / "data/processed").mkdir(parents=True, exist_ok=True)

    L = 220
    thermal_pos = rng.choice(np.arange(10, 100), 30, replace=False)
    pocket_pos = rng.choice(np.arange(140, 215), 24, replace=False)
    p = np.array([0.08, 0.02, 0.05, 0.06, 0.04, 0.07, 0.02, 0.06, 0.06, 0.09, 0.02, 0.04, 0.04, 0.03, 0.05, 0.06, 0.06, 0.07, 0.01, 0.03])
    p /= p.sum()

    rows, ogt_rows, cleave, pockets = [], [], [], []
    taxid = 100000
    for c in range(a.clades):
        fam = FAMS[c % 3]
        root_seq = list(rng.choice(list(AA), L, p=p))
        m = MOTIF[fam]
        root_seq[118 : 118 + len(m)] = list(m)
        base_ogt = float(rng.choice([25, 37, 45, 55, 65, 75, 85], p=[.15, .25, .15, .15, .15, .1, .05]))
        phylum = ["Bacillota", "Pseudomonadota", "Actinomycetota", "Thermotogota", "Euryarchaeota"][c % 5]
        # per-clade preferred residues at P4..P2'
        pref = rng.choice(list(AA), size=(6,), p=p)
        pref[3] = rng.choice(list("KRFLYAV"))
        for k in range(a.members):
            taxid += 1
            ogt = float(np.clip(base_ogt + rng.normal(0, 9), 8, 105))
            s = root_seq.copy()
            for i in range(L):
                if 118 <= i < 118 + len(m):
                    continue
                if rng.random() < 0.12:
                    s[i] = rng.choice(list(AA), p=p)
            p_charged = np.clip(0.12 + 0.55 * (ogt - 30) / 70, 0.05, 0.75)
            for i in thermal_pos:
                s[i] = rng.choice(CHARGED) if rng.random() < p_charged else rng.choice(UNCHARGED)
            bulk_p = np.clip(0.25 + a.tradeoff * 0.5 * (ogt - 30) / 70 + rng.normal(0, 0.07), 0.02, 0.95)
            for i in pocket_pos:
                s[i] = rng.choice(BULKY) if rng.random() < bulk_p else rng.choice(SMALL)
            seq = "".join(s)
            acc = f"SYN{c:03d}_{k:02d}"
            nuc = 118 + SER_OFF[fam] + 1
            rows.append(dict(accession=acc, family=fam, reviewed=False, name=f"synthetic {fam}", taxid=taxid,
                             organism=f"Synthetica {fam.lower()}{c}x strain{k}", lineage=f"Bacteria (superkingdom), {phylum} (phylum)",
                             length=len(seq), sequence=seq, act_sites=str(nuc), nuc_pos=nuc, nuc_source="uniprot", pdb="", phylum=phylum))
            pockets.append(dict(accession=acc, positions=",".join(str(int(i) + 1) for i in sorted(pocket_pos))))
            ogt_rows.append(dict(taxid=taxid, organism=f"Synthetica {fam.lower()}{c}x strain{k}", ogt=round(ogt, 1)))
            # specificity: narrower when pocket is bulky; label ~40% of proteins
            if rng.random() < 0.4:
                sharp = 1.0 + 7.0 * (bulk_p - 0.2)
                n = int(rng.integers(30, 120))
                prof = np.zeros((6, 20))
                for q in range(6):
                    w = np.full(20, 0.6)
                    w[AA.index(pref[q])] += 6.0 * sharp
                    prof[q] = w / w.sum()
                for _ in range(n):
                    win = "".join(rng.choice(list(AA), p=prof[q]) for q in range(6)) + "".join(rng.choice(list(AA), size=2))
                    cleave.append(dict(protease_acc=acc, window=win))
    pd.DataFrame(rows).to_csv(root / "data/processed/sequences.tsv", sep="\t", index=False)
    pd.DataFrame(pockets).to_csv(root / "data/raw/pockets_truth.tsv", sep="\t", index=False)
    pd.DataFrame(ogt_rows).to_csv(root / "data/raw/ogt_table.csv", index=False)
    pd.DataFrame(cleave).to_csv(root / "data/raw/cleavages.tsv", sep="\t", index=False)
    print(f"synthetic: {len(rows)} proteins, {len(cleave)} cleavage windows, tradeoff={a.tradeoff} -> {root}")


if __name__ == "__main__":
    main()
