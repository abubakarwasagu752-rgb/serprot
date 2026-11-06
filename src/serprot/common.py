"""Shared helpers (no torch import here so the data scripts stay light)."""
from __future__ import annotations

import argparse
import math
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}

# Catalytic-Ser fallback motifs (used only when UniProt has no ACT_SITE annotation). Heuristic.
# value = (regex, offset of the nucleophilic Ser inside the match)
SER_MOTIFS = {
    "S8": (re.compile(r"G[TS]S[MILV]A[AS]P"), 2),
    "S1": (re.compile(r"[DGN]S[GAC]GP"), 1),
    "S9": (re.compile(r"G[WGAHF]SYG[GA]"), 2),
}


def base_parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--config", default="config.yaml")
    return ap


def load_config(path: str | Path) -> dict:
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def rpath(cfg: dict, key: str, *parts: str, mkdir: bool = True) -> Path:
    """Resolve cfg['paths'][key]/parts under the project root; create the directory part."""
    root = Path(cfg["paths"]["root"]).expanduser()
    p = (root / cfg["paths"][key]).joinpath(*parts)
    if mkdir:
        (p.parent if p.suffix else p).mkdir(parents=True, exist_ok=True)
    return p


def cfg_file(cfg: dict, rel: str) -> Path:
    """A file given relative to the project root in the config (e.g. ogt.table)."""
    return Path(cfg["paths"]["root"]).expanduser() / rel


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def parse_phylum(lineage: str) -> str:
    if not isinstance(lineage, str):
        return "Unknown"
    m = re.search(r"([^,]+?) \(phylum\)", lineage)
    if m:
        return m.group(1).strip()
    m = re.search(r"([^,]+?) \(clade\)", lineage)
    return m.group(1).strip() if m else "Unknown"


def norm_species(name: str) -> str:
    """'Thermus thermophilus HB8' -> 'thermus thermophilus' (binomial, lowercase)."""
    if not isinstance(name, str):
        return ""
    name = re.sub(r"\(.*?\)", " ", name)
    toks = re.sub(r"[^A-Za-z. ]", " ", name).split()
    return " ".join(toks[:2]).lower()


def find_nucleophile(seq: str, family: str, act_sites: list[int]) -> tuple[int | None, str]:
    """Return (1-based position of the catalytic Ser, source)."""
    sers = [p for p in act_sites if 1 <= p <= len(seq) and seq[p - 1] == "S"]
    if len(sers) == 1:
        return sers[0], "uniprot"
    if len(sers) > 1:
        return sers[-1], "uniprot"
    rx, off = SER_MOTIFS.get(family, (None, 0))
    if rx is not None:
        m = rx.search(seq)
        if m:
            return m.start() + off + 1, "motif"
    return None, "none"


def crop_window(seq: str, anchors: list[int], max_len: int) -> tuple[str, int]:
    """Crop to max_len residues centred on the catalytic residues; returns (seq, 0-based offset)."""
    n = len(seq)
    if n <= max_len:
        return seq, 0
    anchors = [a - 1 for a in anchors if a]
    centre = int(np.mean(anchors)) if anchors else n // 2
    start = int(max(0, min(centre - max_len // 2, n - max_len)))
    return seq[start : start + max_len], start


# ------------------------------------------------------------------ specificity profiles
def profile_breadth(probs: np.ndarray) -> np.ndarray:
    """Mean normalised Shannon entropy over positions. probs: [..., P, 20] -> [...]. 0 = strict, 1 = flat."""
    p = np.clip(probs, 1e-12, 1.0)
    h = -(p * np.log(p)).sum(-1) / math.log(20)
    return h.mean(-1)


def profile_p1_breadth(probs: np.ndarray, p1_index: int) -> np.ndarray:
    p = np.clip(probs[..., p1_index, :], 1e-12, 1.0)
    return -(p * np.log(p)).sum(-1) / math.log(20)


def window_counts(windows: list[str], positions: list[int]) -> np.ndarray:
    """Count residues per window position. Non-standard letters and gaps are skipped."""
    c = np.zeros((len(positions), 20), dtype=np.float64)
    for w in windows:
        if not isinstance(w, str) or len(w) < max(positions) + 1:
            continue
        for k, pos in enumerate(positions):
            j = AA_IDX.get(w[pos].upper())
            if j is not None:
                c[k, j] += 1
    return c


def counts_to_profile(counts: np.ndarray, pseudocount: float) -> np.ndarray:
    c = counts + pseudocount
    return c / c.sum(-1, keepdims=True)


# ------------------------------------------------------------------ metrics
def safe_corr(a, b, method="pearson") -> float:
    from scipy import stats

    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return float("nan")
    f = stats.pearsonr if method == "pearson" else stats.spearmanr
    return float(f(a[m], b[m])[0])


def load_dataset(cfg: dict) -> pd.DataFrame:
    p = rpath(cfg, "processed", "dataset.tsv")
    return pd.read_csv(p, sep="\t")


def prepare_inputs(df: pd.DataFrame, max_len: int) -> pd.DataFrame:
    """Add seq_in (cropped to max_len around the catalytic site) and crop_off (0-based offset)."""
    df = df.copy()
    seqs, offs = [], []
    for s, nuc, act in zip(df["sequence"], df["nuc_pos"], df["act_sites"].fillna("")):
        anchors = [int(x) for x in str(act).split(",") if x.strip().isdigit()]
        if not anchors and pd.notna(nuc):
            anchors = [int(nuc)]
        c, o = crop_window(s, anchors, max_len)
        seqs.append(c)
        offs.append(o)
    df["seq_in"], df["crop_off"] = seqs, offs
    return df


def load_profiles(cfg: dict) -> dict[str, np.ndarray]:
    """accession -> smoothed profile [P,20] (empty dict if no cleavage data)."""
    p = rpath(cfg, "processed", "profiles.npz")
    if not Path(p).exists():
        return {}
    z = np.load(p, allow_pickle=True)
    probs = counts_to_profile(z["counts"], cfg["specificity"]["pseudocount"])
    return {a: probs[i] for i, a in enumerate(z["accs"])}
