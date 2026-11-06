"""Within-clade regression helpers, shared so every stage computes slopes the same way."""
from __future__ import annotations

import numpy as np


def demean(v: np.ndarray, codes: np.ndarray, k: int) -> np.ndarray:
    """Subtract the clade mean (clade fixed effects)."""
    sums = np.bincount(codes, weights=v, minlength=k)
    cnts = np.bincount(codes, minlength=k).astype(float)
    return v - (sums / np.maximum(cnts, 1))[codes]


def slope(y: np.ndarray, x: np.ndarray, codes: np.ndarray, k: int, z: np.ndarray | None = None) -> float:
    """Within-clade slope of y on x, expressed per +10 units of x. z is partialled out if given."""
    ok = np.isfinite(y) & np.isfinite(x)
    if z is not None:
        ok &= np.isfinite(z)
    if ok.sum() < 5:
        return np.nan
    yc, xc = demean(y[ok], codes[ok], k), demean(x[ok], codes[ok], k)
    if z is not None:
        zc = demean(z[ok], codes[ok], k)
        zz = (zc * zc).sum()
        if zz > 0:
            yc = yc - zc * (zc * yc).sum() / zz
            xc = xc - zc * (zc * xc).sum() / zz
    den = (xc * xc).sum()
    return float((xc * yc).sum() / den * 10) if den > 0 else np.nan


def within_sd(y: np.ndarray, codes: np.ndarray, k: int) -> float:
    ok = np.isfinite(y)
    if ok.sum() < 3:
        return np.nan
    return float(demean(y[ok], codes[ok], k).std())


def clade_bootstrap(fn, codes: np.ndarray, k: int, n_boot: int, rng):
    """Resample whole clades with replacement; fn(idx, recoded) -> statistic."""
    members = [np.flatnonzero(codes == i) for i in range(k)]
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, k, k)
        idx = np.concatenate([members[i] for i in pick])
        rc = np.repeat(np.arange(k), [len(members[i]) for i in pick])
        v = fn(idx, rc)
        if np.isfinite(v):
            out.append(v)
    return np.array(out)


def permute_within_clades(x: np.ndarray, codes: np.ndarray, k: int, rng) -> np.ndarray:
    """Shuffle x inside each clade, preserving clade structure and the marginal distribution."""
    xs = x.copy()
    for i in range(k):
        m = np.flatnonzero(codes == i)
        if len(m) > 1:
            xs[m] = rng.permutation(xs[m])
    return xs
