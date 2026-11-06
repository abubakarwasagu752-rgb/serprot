#!/usr/bin/env python
"""Refold proteins with ESMFold, so the pocket analysis can be repeated on structures from ONE predictor.

Why: every structure so far came from AlphaFold DB, which covered only ~61% of targets and whose
coverage is biased by accession vintage (and therefore, indirectly, by temperature). Folding here
removes both problems at once -- one model, one version, no coverage gap -- at the cost of somewhat
lower accuracy than AlphaFold2. Within-clade contrasts are comparatively forgiving of that, because
systematic error largely cancels between close relatives.

Selection prioritises clades that the analysis can actually use (several OGT-labelled members spanning
a real temperature range), and folds ALL their members, including the ones AlphaFold DB lacks.

  python scripts/17_esmfold.py --config config.yaml --limit 20          # smoke test first
  python scripts/17_esmfold.py --config config.yaml --shard $SLURM_ARRAY_TASK_ID --n-shards 4
"""
import gzip
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serprot.common import base_parser, load_config, load_dataset, rpath, seed_all  # noqa: E402


def select(cfg, min_members, min_spread, max_len, budget, seed, min_len=0):
    df = load_dataset(cfg)
    df = df[df["nuc_pos"].notna() & df["ogt"].notna()].copy()
    df["ogt"] = df["ogt"].astype(float)
    # clade membership is decided on the FULL set, then members are filtered by length, so a
    # length window never silently changes which clades qualify
    g0 = df.groupby("cluster_pairs")["ogt"]
    eligible = g0.size().ge(min_members) & (g0.max() - g0.min()).ge(min_spread)
    df = df[df["cluster_pairs"].isin(eligible[eligible].index)]
    df = df[(df["length"] <= max_len) & (df["length"] >= min_len)]
    g = df.groupby("cluster_pairs")["ogt"]
    stats = pd.DataFrame({"n": g.size(), "spread": g.max() - g.min(),
                          "hot": df.assign(h=df["ogt"] >= 50).groupby("cluster_pairs")["h"].sum()})
    keep = stats[stats["n"] >= 1]
    # spend the budget on the most informative clades first: wide temperature range, thermophiles present
    keep = keep.assign(score=keep["spread"] + 10 * (keep["hot"] > 0)).sort_values("score", ascending=False)
    chosen, total = [], 0
    for cl in keep.index:
        members = df[df["cluster_pairs"] == cl]
        if budget and total + len(members) > budget:
            continue
        chosen.append(members)
        total += len(members)
    sel = pd.concat(chosen).sort_values(["cluster_pairs", "ogt"]) if chosen else df.head(0)
    return sel, len(keep)


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--min-members", type=int, default=3)
    ap.add_argument("--min-spread", type=float, default=10.0)
    ap.add_argument("--min-len", type=int, default=0,
                    help="skip proteins shorter than this; use with --max-len to split long and short "
                         "chains across GPUs of different memory")
    ap.add_argument("--max-len", type=int, default=750,
                    help="ESMFold memory grows steeply with length; 700 is comfortable on a 46 GB L40")
    ap.add_argument("--budget", type=int, default=0, help="max proteins to fold (0 = no cap)")
    ap.add_argument("--chunk-size", type=int, default=64, help="lower (e.g. 32) if you hit OOM")
    ap.add_argument("--limit", type=int, default=0, help="fold only this many -- use for a smoke test")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--out-dir", default="data/structures_esmfold")
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed_all(cfg["seed"])

    sel, n_clades = select(cfg, args.min_members, args.min_spread, args.max_len, args.budget,
                           cfg["seed"], args.min_len)
    outdir = Path(cfg["paths"]["root"]).expanduser() / args.out_dir
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"length window {args.min_len}-{args.max_len} residues")
    print(f"{n_clades} clades pass (>={args.min_members} labelled members, >={args.min_spread} C spread); "
          f"folding {len(sel)} proteins in {sel['cluster_pairs'].nunique()} clades "
          f"({int((sel['ogt'] >= 50).sum())} thermophilic)")

    todo = sel.iloc[args.shard :: args.n_shards]
    if args.limit:
        todo = todo.head(args.limit)
    todo = [r for r in todo.itertuples() if not (outdir / f"{r.accession}.pdb.gz").exists()]
    print(f"shard {args.shard}/{args.n_shards}: {len(todo)} still to fold", flush=True)
    if not todo:
        print("nothing to do")
        return

    import torch
    from transformers import AutoTokenizer, EsmForProteinFolding

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    model = model.to(dev).eval()
    if dev.type == "cuda":
        model.esm = model.esm.half()                 # language-model half of ESMFold in fp16
        torch.backends.cuda.matmul.allow_tf32 = True
    try:
        model.trunk.set_chunk_size(args.chunk_size)  # trades speed for memory on long chains
    except Exception as e:  # noqa: BLE001
        print(f"could not set chunk size ({e}); continuing with the default")
    print(f"ESMFold loaded on {dev}", flush=True)

    t0, done, failed = time.time(), 0, 0
    for r in todo:
        seq = str(r.sequence).upper().replace("U", "C").replace("O", "K")
        seq = "".join(c for c in seq if c in "ACDEFGHIKLMNPQRSTVWY")
        if len(seq) < 50:
            failed += 1
            continue
        try:
            with torch.no_grad():
                pdb = model.infer_pdb(seq)
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM on {r.accession} (len {len(seq)}); skipping", flush=True)
            torch.cuda.empty_cache()
            failed += 1
            continue
        except Exception as e:  # noqa: BLE001
            print(f"  failed {r.accession}: {type(e).__name__}: {e}", flush=True)
            failed += 1
            continue
        tmp = outdir / f"{r.accession}.tmp"
        tmp.write_bytes(gzip.compress(pdb.encode()))
        tmp.rename(outdir / f"{r.accession}.pdb.gz")
        done += 1
        if done % 25 == 0:
            el = time.time() - t0
            print(f"  {done}/{len(todo)}  {el / done:.1f}s each  "
                  f"eta {(len(todo) - done) * el / done / 3600:.1f} h  failed={failed}", flush=True)

    el = time.time() - t0
    have = len(list(outdir.glob("*.pdb.gz")))
    print(f"\nfolded {done} in {el / 60:.1f} min ({el / max(done, 1):.1f} s each); failed {failed}")
    print(f"structures in {outdir}: {have}")
    if done:
        # ESMFold writes per-residue pLDDT into the B-factor column, same convention as AlphaFold
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from serprot.structure import read_pdb
        sample = list(outdir.glob("*.pdb.gz"))[:20]
        pl = [float(np.mean(read_pdb(f).plddt)) for f in sample]
        print(f"mean pLDDT over {len(pl)} sampled models: {np.mean(pl):.1f} "
              f"(range {min(pl):.1f}-{max(pl):.1f})")
        print("compare with the AlphaFold set, whose pocket pLDDT median was 94.9")


if __name__ == "__main__":
    main()
