"""Minimal, fast reader for AlphaFold PDB models plus the geometry needed for pocket descriptors.

AlphaFold models are single-chain, one altloc, no heteroatoms, so a purpose-built parser is both
faster and more predictable than a general one. B-factor carries the per-residue pLDDT.
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Kyte-Doolittle hydropathy
KD = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
      "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}
THREE2ONE = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
             "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
             "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
# van der Waals radii (A); used for SASA
VDW = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "H": 1.20}
CHARGED_POS, CHARGED_NEG, AROMATIC = set("KR"), set("DE"), set("FWY")
# residue volumes (A^3), Zamyatnin 1972
RESVOL = {"A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8, "E": 138.4, "G": 60.1,
          "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7, "S": 89.0,
          "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0}
POLAR, SMALL = set("STNQHCY"), set("GAS")


@dataclass
class Structure:
    resnum: np.ndarray        # [R] author residue numbers
    aa: np.ndarray            # [R] one-letter codes
    plddt: np.ndarray         # [R] per-residue pLDDT (B-factor of CA)
    coords: np.ndarray        # [A, 3] all heavy-atom coordinates
    atom_res: np.ndarray      # [A] index into resnum for each atom
    atom_elem: np.ndarray     # [A] element symbol
    atom_name: np.ndarray     # [A] PDB atom name, stripped
    ca: np.ndarray            # [R, 3] CA coordinates

    def __len__(self) -> int:
        return len(self.resnum)

    def res_index(self, resnum: int) -> int | None:
        hit = np.flatnonzero(self.resnum == resnum)
        return int(hit[0]) if len(hit) else None

    def atom_coord(self, res_i: int, name: str) -> np.ndarray | None:
        sel = (self.atom_res == res_i) & (self.atom_name == name)
        return self.coords[sel][0] if sel.any() else None

    def min_dist_to(self, point: np.ndarray) -> np.ndarray:
        """[R] distance from each residue's nearest heavy atom to `point`."""
        d = np.linalg.norm(self.coords - point, axis=1)
        out = np.full(len(self), np.inf)
        np.minimum.at(out, self.atom_res, d)
        return out


def read_pdb(path: str | Path) -> Structure | None:
    p = Path(path)
    raw = gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()
    text = raw.decode("latin-1")
    resnum, aa, plddt = [], [], []
    coords, atom_res, elem, aname = [], [], [], []
    cur, ri = None, -1
    ca_by_res: dict[int, np.ndarray] = {}
    for line in text.splitlines():
        if not line.startswith("ATOM"):
            if line.startswith("ENDMDL"):
                break
            continue
        alt = line[16]
        if alt not in (" ", "A"):
            continue
        e = line[76:78].strip() or line[12:16].strip()[0]
        if e == "H":
            continue
        rn = int(line[22:26])
        rname = line[17:20].strip()
        if rn != cur:
            cur = rn
            ri += 1
            resnum.append(rn)
            aa.append(THREE2ONE.get(rname, "X"))
            plddt.append(float(line[60:66]))
        nm = line[12:16].strip()
        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        coords.append(xyz)
        atom_res.append(ri)
        elem.append(e)
        aname.append(nm)
        if nm == "CA":
            ca_by_res[ri] = np.array(xyz, dtype=np.float32)
    if ri < 0:
        return None
    ca = np.array([ca_by_res.get(i, np.full(3, np.nan, dtype=np.float32)) for i in range(ri + 1)], dtype=np.float32)
    plddt = np.array(plddt, dtype=np.float32)
    # AlphaFold writes pLDDT as 0-100 in the B-factor column; ESMFold writes 0-1. Rescale the latter so
    # confidence filters (>= 70) mean the same thing for both, instead of silently rejecting every
    # ESMFold model. Genuine 0-100 pLDDT is never entirely <= 1.0, so this test is safe.
    if len(plddt) and np.nanmax(plddt) <= 1.0:
        plddt = plddt * 100.0
    return Structure(np.array(resnum), np.array(aa), plddt,
                     np.array(coords, dtype=np.float32), np.array(atom_res), np.array(elem), np.array(aname), ca)


def sasa(st: Structure, n_sphere: int = 256, probe: float = 1.4, res_idx=None) -> np.ndarray:
    """Shrake-Rupley solvent-accessible surface area per residue (A^2).

    All atoms always act as occluders; `res_idx` restricts which residues the area is computed FOR,
    which is what makes this affordable when only a pocket and a control patch are needed.
    Returns a [R] array with NaN for residues that were not requested.
    """
    from scipy.spatial import cKDTree

    r = np.array([VDW.get(e, 1.7) for e in st.atom_elem], dtype=np.float64) + probe
    k = np.arange(n_sphere, dtype=np.float64) + 0.5
    phi = np.arccos(1 - 2 * k / n_sphere)
    theta = np.pi * (1 + 5**0.5) * k
    sph = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)

    want = np.ones(len(st), dtype=bool)
    if res_idx is not None:
        want[:] = False
        want[np.asarray(list(res_idx), dtype=int)] = True
    atoms_wanted = np.flatnonzero(want[st.atom_res])

    tree = cKDTree(st.coords)
    rmax = float(r.max())
    out_atom = np.zeros(len(st.coords), dtype=np.float64)
    neigh = tree.query_ball_point(st.coords[atoms_wanted], r=rmax + rmax, workers=-1)
    for a, cand in zip(atoms_wanted, neigh):
        cand = np.array([c for c in cand if c != a], dtype=np.int64)
        if len(cand) == 0:
            out_atom[a] = 4 * np.pi * r[a] ** 2
            continue
        # keep only atoms whose sphere can actually reach this one
        d = np.linalg.norm(st.coords[cand] - st.coords[a], axis=1)
        cand = cand[d < r[a] + r[cand]]
        if len(cand) == 0:
            out_atom[a] = 4 * np.pi * r[a] ** 2
            continue
        pts = st.coords[a] + r[a] * sph
        d2 = ((pts[:, None, :] - st.coords[cand][None, :, :]) ** 2).sum(-1)
        frac = 1.0 - (d2 < (r[cand] ** 2)[None, :]).any(axis=1).mean()
        out_atom[a] = 4 * np.pi * r[a] ** 2 * frac

    out = np.full(len(st), np.nan, dtype=np.float32)
    out[want] = 0.0
    np.add.at(out, st.atom_res[atoms_wanted], out_atom[atoms_wanted])
    return out


def neighbor_count(st: Structure, radius: float = 10.0) -> np.ndarray:
    """[R] heavy atoms (of other residues) within `radius` of each residue's CA -- a cheap burial proxy."""
    from scipy.spatial import cKDTree

    ok = ~np.isnan(st.ca[:, 0])
    out = np.zeros(len(st), dtype=np.float32)
    tree = cKDTree(st.coords)
    idx = tree.query_ball_point(st.ca[ok], r=radius, workers=-1)
    own = st.atom_res
    out[ok] = [sum(1 for j in lst if own[j] != i) for i, lst in zip(np.flatnonzero(ok), idx)]
    return out
