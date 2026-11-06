#!/usr/bin/env python
"""Synthetic AlphaFold-like models with a PLANTED, known temperature signal, to test stages 11-12.

Two effects are planted deliberately:
  global  : charged-residue fraction rises with OGT EVERYWHERE (the real, well-known thermal signal)
  pocket  : bulky/aromatic fraction rises with OGT ONLY in the pocket (strength --pocket-effect)
A correct analysis must recover the pocket effect as pocket-vs-control, and must NOT report the
global charge effect as pocket-specific.
"""
import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

THREE = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU", "G": "GLY",
         "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO", "S": "SER",
         "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL"}
BULKY, PLAIN, CHARGED = list("FWYLIM"), list("GASTV"), list("EKRD")
P_BULKY_BASE = 0.25          # background bulky rate OUTSIDE the pocket (and inside it when effect = 0)


def write_model(path, aa, ca, nuc_i, plddt):
    lines, i = ["HEADER    SYNTH"], 0
    rng = np.random.default_rng(abs(hash(str(path))) % 2**31)
    for ri, (a, c) in enumerate(zip(aa, ca)):
        names = [("N", "N"), ("CA", "C"), ("C", "C"), ("O", "O")]
        if a != "G":
            names.append(("CB", "C"))
        if ri == nuc_i:
            names.append(("OG", "O"))          # the catalytic serine's hydroxyl
        elif a in BULKY:
            names += [("CG", "C"), ("CD", "C")]
        for k, (nm, el) in enumerate(names):
            i += 1
            off = np.zeros(3) if nm == "CA" else rng.normal(scale=1.3, size=3)
            x, y, z = c + off
            lines.append(f"ATOM  {i:>5} {nm:<4} {THREE[a]:>3} A{ri + 1:>4}    "
                         f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.0:>6.2f}{plddt[ri]:>6.2f}          {el:>2}")
    lines.append("END")
    Path(path).write_bytes(gzip.compress(("\n".join(lines) + "\n").encode()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--clades", type=int, default=20)
    ap.add_argument("--members", type=int, default=6)
    ap.add_argument("--pocket-effect", type=float, default=1.0, help="0 = pocket behaves exactly like the rest")
    ap.add_argument("--shield-pocket", action="store_true",
                    help="pocket composition is INDEPENDENT of OGT while the rest of the protein adapts "
                         "-- i.e. the pattern the real data shows; used to test detection")
    ap.add_argument("--flex-effect", type=float, default=0.0,
                    help="plant a DYNAMIC effect for stage 18: pocket residues are pushed outward in "
                         "proportion to OGT, thinning their contacts and raising their GNM fluctuation, "
                         "with composition untouched. 0 = no dynamic effect (the null stage 18 must return)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    root = Path(a.root)
    sdir = root / "data/structures"
    sdir.mkdir(parents=True, exist_ok=True)
    (root / "data/processed").mkdir(parents=True, exist_ok=True)

    N = 160
    rows = []
    for c in range(a.clades):
        fam = ["S8", "S9", "S1"][c % 3]
        # one fold per clade: positions are shared by all members, so pockets are comparable
        pos = rng.normal(size=(N, 3))
        pos /= np.linalg.norm(pos, axis=1, keepdims=True)
        pos *= (rng.random((N, 1)) ** (1 / 3)) * 15.0
        nuc_i = int(np.argmax(np.linalg.norm(pos, axis=1)))        # on the surface
        d_nuc = np.linalg.norm(pos - pos[nuc_i], axis=1)
        pocket = d_nuc <= 10.0
        far = d_nuc > 15.0
        base_ogt = float(rng.choice([25, 35, 45, 55, 70]))
        for m in range(a.members):
            ogt = float(np.clip(base_ogt + rng.normal(0, 12), 5, 100))
            p_charged = np.clip(0.10 + 0.25 * (ogt - 20) / 70, 0.02, 0.8)     # global effect
            p_bulky = np.clip(P_BULKY_BASE + a.pocket_effect * 0.5 * (ogt - 20) / 70, 0.02, 0.95)  # pocket only
            # charge is the GLOBAL effect: one draw, identical odds inside and outside the pocket.
            # among the residues that did not become charged, only the pocket gets the bulky trend.
            aa = []
            for i in range(N):
                if i == nuc_i:
                    aa.append("S")
                elif not (a.shield_pocket and pocket[i]) and rng.random() < p_charged:
                    aa.append(rng.choice(CHARGED))
                elif a.shield_pocket and pocket[i]:
                    # pocket drawn from a FIXED distribution: no temperature information at all
                    aa.append(rng.choice(BULKY) if rng.random() < P_BULKY_BASE else rng.choice(PLAIN))
                else:
                    # identical machinery inside and outside; ONLY the bulky probability differs,
                    # so pocket-effect = 0 is a true null (pocket and control are exchangeable)
                    pb = p_bulky if pocket[i] else P_BULKY_BASE
                    aa.append(rng.choice(BULKY) if rng.random() < pb else rng.choice(PLAIN))
            acc = f"SY{c:03d}{m:02d}"
            plddt = np.clip(rng.normal(88, 6, N), 40, 99)
            coords = pos + rng.normal(scale=0.25, size=pos.shape)
            if a.flex_effect:
                # a purely GEOMETRIC thermal signal: pocket residues drift outward with OGT, so their
                # local contact density falls and their network fluctuation rises. Composition is
                # untouched, so stage 12 must stay null here while stage 18 must fire.
                v = coords[pocket] - pos[nuc_i]
                nrm = np.linalg.norm(v, axis=1, keepdims=True)
                nrm[nrm == 0] = 1.0
                coords[pocket] = coords[pocket] + (v / nrm) * (a.flex_effect * 0.06 * (ogt - 20))
            write_model(sdir / f"{acc}.pdb.gz", aa, coords, nuc_i, plddt)
            rows.append(dict(accession=acc, family=fam, cluster_pairs=f"CL{c:03d}", ogt=round(ogt, 1),
                             nuc_pos=nuc_i + 1, length=N, organism=f"Synth {c}-{m}",
                             true_pocket_bulky=p_bulky, true_charged=p_charged))
    tg = pd.DataFrame(rows)
    tg.to_csv(root / "data/processed/structure_targets.tsv", sep="\t", index=False)
    print(f"{len(tg)} models in {a.clades} clades -> {sdir}  (pocket-effect={a.pocket_effect})")


if __name__ == "__main__":
    main()
