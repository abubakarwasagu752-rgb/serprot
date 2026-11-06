"""Pocket and control-patch selection, shared so every stage selects residues identically.

The logic is lifted verbatim from 11_pocket_descriptors.py; `regions_for()` must keep returning the
same indices that stage produced, or the masking experiment would not be testing the same residues.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .structure import Structure, neighbor_count


def anchor_point(st: Structure, nuc_pos: int):
    """-> (residue index, coordinate, how it was found) or (None, None, reason)."""
    ri = st.res_index(int(nuc_pos))
    if ri is None:
        return None, None, "nuc_pos_not_in_model"
    og = st.atom_coord(ri, "OG")
    if og is not None:
        return ri, og, "OG"
    ca = st.ca[ri]
    if np.isnan(ca).any():
        return None, None, "no_coords_at_nuc_pos"
    return ri, ca, "CA_fallback"


def regions_for(st: Structure, nuc_pos: int, acc: str, radius: float = 10.0,
                n_controls: int = 8, ctrl_min_dist: float = 15.0):
    """-> dict(pocket=[idx], controls=[[idx], ...], anchor_res=i, how=str) or None."""
    ri, pt, how = anchor_point(st, nuc_pos)
    if ri is None:
        return None
    dist = st.min_dist_to(pt)
    nc = neighbor_count(st)
    pocket = np.flatnonzero(dist <= radius)
    if len(pocket) < 5:
        return None

    rng = np.random.default_rng(int(hashlib.sha1(acc.encode()).hexdigest()[:8], 16))
    far = np.flatnonzero(dist > ctrl_min_dist)
    controls = []
    if len(far):
        order = far[np.argsort(np.abs(nc[far] - nc[ri]))]
        pool = order[: max(len(order) // 2, min(20, len(order)))]
        for c in rng.choice(pool, size=min(n_controls, len(pool)), replace=False):
            dc = np.linalg.norm(st.ca - st.ca[c], axis=1)
            dc[np.isnan(dc)] = np.inf
            controls.append(np.argsort(dc)[: len(pocket)])
    return {"pocket": pocket, "controls": controls, "anchor_res": ri, "how": how}


def to_sequence_positions(st: Structure, idx: np.ndarray, crop_off: int, seq_len: int) -> np.ndarray:
    """Residue indices in the model -> 0-based positions in the cropped sequence fed to the model.

    AlphaFold numbers residues by UniProt position, and prepare_inputs() crops `crop_off` residues
    off the front, so the mapping is resnum - 1 - crop_off. Positions outside the crop are dropped.
    """
    pos = st.resnum[idx].astype(int) - 1 - int(crop_off)
    return pos[(pos >= 0) & (pos < seq_len)]
