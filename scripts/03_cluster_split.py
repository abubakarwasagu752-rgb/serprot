#!/usr/bin/env python
"""Cluster sequences and build lineage-aware splits.

Adds to labels.tsv -> dataset.tsv:
  cluster_split  (identity = clustering.id_split)  groups for held-out evaluation
  cluster_pairs  (identity = clustering.id_pairs)  clades for within-clade trade-off tests
  fold_cv        0..n_folds-1  StratifiedGroupKFold by cluster_split, stratified by family
  fold_lofo      index of the family held out (leave-one-family-out)

Uses MMseqs2 when available. The pure-python fallback (k-mer cosine, greedy) is only for tests and small data.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serprot.common import AA_IDX, base_parser, load_config, rpath, seed_all  # noqa: E402


def mmseqs_cluster(df: pd.DataFrame, ident: float, cov: float, threads: int) -> pd.Series:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fa = td / "seqs.fa"
        with open(fa, "w") as f:
            for a, s in zip(df["accession"], df["sequence"]):
                f.write(f">{a}\n{s}\n")
        cmd = ["mmseqs", "easy-cluster", str(fa), str(td / "out"), str(td / "tmp"),
               "--min-seq-id", str(ident), "-c", str(cov), "--cov-mode", "1", "--threads", str(threads), "-v", "1"]
        subprocess.run(cmd, check=True)
        cl = pd.read_csv(td / "out_cluster.tsv", sep="\t", header=None, names=["rep", "member"])
    rep = cl.set_index("member")["rep"]
    return pd.Series(df["accession"].map(rep).fillna(df["accession"]).values)


def kmer_matrix(seqs, k=3):
    idx = {}
    m = np.zeros((len(seqs), 20**k), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j in range(len(s) - k + 1):
            code = 0
            ok = True
            for ch in s[j : j + k]:
                a = AA_IDX.get(ch)
                if a is None:
                    ok = False
                    break
                code = code * 20 + a
            if ok:
                m[i, code] += 1
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.maximum(n, 1e-9)


def fallback_cluster(df: pd.DataFrame, ident: float) -> pd.Series:
    """Greedy k-mer cosine clustering. Crude: threshold is a monotone proxy for identity, NOT identity."""
    print(f"  [fallback clustering at proxy identity {ident}; install mmseqs2 for real runs]")
    thr = {0.3: 0.20, 0.5: 0.29}.get(round(ident, 2), 0.2 + 0.4 * ident)
    X = kmer_matrix(df["sequence"].tolist())
    order = np.argsort(-df["sequence"].str.len().values)
    rep_of = np.full(len(df), -1)
    reps: list[int] = []
    for i in order:
        if reps:
            sims = X[reps] @ X[i]
            j = int(np.argmax(sims))
            if sims[j] >= thr:
                rep_of[i] = reps[j]
                continue
        reps.append(i)
        rep_of[i] = i
    return pd.Series(df["accession"].values[rep_of])


def cluster(df, ident, cov, threads):
    if shutil.which("mmseqs"):
        return mmseqs_cluster(df, ident, cov, threads)
    return fallback_cluster(df, ident)


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_all(cfg["seed"])
    df = pd.read_csv(rpath(cfg, "processed", "labels.tsv"), sep="\t")
    cc = cfg["clustering"]

    df["cluster_split"] = cluster(df, cc["id_split"], cc["coverage"], args.threads).values
    df["cluster_pairs"] = cluster(df, cc["id_pairs"], cc["coverage"], args.threads).values
    print(f"clusters: split={df['cluster_split'].nunique()}  pairs={df['cluster_pairs'].nunique()}  of {len(df)} sequences")

    from sklearn.model_selection import StratifiedGroupKFold

    sgkf = StratifiedGroupKFold(n_splits=cc["n_folds"], shuffle=True, random_state=cfg["seed"])
    df["fold_cv"] = -1
    for k, (_, te) in enumerate(sgkf.split(df, df["family"], groups=df["cluster_split"])):
        df.loc[df.index[te], "fold_cv"] = k
    fams = list(cfg["families"].keys())
    df["fold_lofo"] = df["family"].map({f: i for i, f in enumerate(fams)})

    # sanity: no cluster may straddle folds
    straddle = df.groupby("cluster_split")["fold_cv"].nunique().gt(1).sum()
    assert straddle == 0, f"{straddle} clusters straddle folds"
    out = rpath(cfg, "processed", "dataset.tsv")
    df.to_csv(out, sep="\t", index=False)
    print(df.groupby("fold_cv").agg(n=("accession", "size"), ogt=("ogt", "count"), spec=("spec_n", "count")).to_string())
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
