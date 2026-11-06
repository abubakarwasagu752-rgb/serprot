#!/usr/bin/env python
"""What fraction of our targets actually exist in AlphaFold DB, and is the gap correlated with temperature?

Samples accessions at RANDOM (not in file order, which is biased toward ones already tried), asks the API
once per accession, and breaks coverage down by family, OGT band and accession vintage.

  python scripts/diag_afdb_coverage.py --config config.yaml --n 400 --workers 4
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serprot.common import base_parser, load_config, rpath  # noqa: E402

API = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"
UA = {"User-Agent": "serprot-pipeline (academic research)"}


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--stratify-ogt", action="store_true",
                    help="sample hot and cool targets equally, so thermophile coverage is actually measurable")
    ap.add_argument("--hot-cut", type=float, default=50.0)
    ap.add_argument("--workers", type=int, default=4, help="keep low: 8+ threads triggered 429s here")
    args = ap.parse_args()
    cfg = load_config(args.config)

    tg = pd.read_csv(rpath(cfg, "processed", "structure_targets.tsv"), sep="\t")
    rng = np.random.default_rng(cfg["seed"])
    if args.stratify_ogt:
        # thermophiles are ~4% of the set, so a plain random sample cannot measure them.
        # Take equal numbers of hot and cool targets; coverage per stratum is still unbiased.
        hot = tg[tg["ogt"] >= args.hot_cut]
        cool = tg[tg["ogt"] < args.hot_cut]
        k = min(args.n // 2, len(hot))
        samp = pd.concat([hot.iloc[rng.choice(len(hot), size=k, replace=False)],
                          cool.iloc[rng.choice(len(cool), size=min(args.n - k, len(cool)), replace=False)]]).copy()
        print(f"STRATIFIED sample: {k} targets with OGT >= {args.hot_cut} C and {len(samp) - k} below, "
              f"out of {len(hot)} and {len(cool)} available, {args.workers} threads\n")
    else:
        samp = tg.iloc[rng.choice(len(tg), size=min(args.n, len(tg)), replace=False)].copy()
        print(f"sampling {len(samp)} of {len(tg)} targets at random, {args.workers} threads\n")

    session = requests.Session()

    def check(acc):
        for attempt in range(4):
            try:
                r = session.get(API.format(acc=acc), timeout=45, headers=UA)
            except Exception:  # noqa: BLE001
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 2 * (attempt + 1))))
                continue
            if r.status_code == 404:
                return 0
            if r.status_code == 200:
                try:
                    return 1 if r.json() else 0
                except Exception:  # noqa: BLE001
                    return -1
            time.sleep(1 + attempt)
        return -1

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        samp["has_model"] = list(ex.map(check, samp["accession"]))
    el = time.time() - t0
    undecided = int((samp["has_model"] < 0).sum())
    ok = samp[samp["has_model"] >= 0]
    print(f"{len(ok)} decided in {el:.0f}s ({len(samp) / el:.1f}/s); {undecided} undecided\n")

    if args.stratify_ogt:
        print("(sample is stratified, so the OVERALL figure below is NOT the population coverage; "
              "read the per-stratum rows instead)")
    cov = ok["has_model"].mean()
    lo, hi = cov - 1.96 * np.sqrt(cov * (1 - cov) / len(ok)), cov + 1.96 * np.sqrt(cov * (1 - cov) / len(ok))
    print(f"OVERALL COVERAGE: {cov:.1%}  (95% CI {lo:.1%}-{hi:.1%})")
    print(f"  => of {len(tg)} targets, expect ~{int(cov * len(tg)):,} models\n")

    print("by family:")
    print(ok.groupby("family")["has_model"].agg(n="size", coverage="mean").assign(
        coverage=lambda d: (d.coverage * 100).round(1)).to_string())

    ok = ok.assign(ogt_band=pd.cut(ok["ogt"], [0, 25, 35, 45, 55, 120],
                                   labels=["<25", "25-35", "35-45", "45-55", ">=55"]))
    print("\nby growth temperature  (THIS IS THE ONE THAT MATTERS -- a trend here is a confound):")
    print(ok.groupby("ogt_band", observed=True)["has_model"].agg(n="size", coverage="mean").assign(
        coverage=lambda d: (d.coverage * 100).round(1)).to_string())

    # accession vintage: A0A.. entries are recent TrEMBL additions, mostly post-dating AlphaFold DB
    ok = ok.assign(vintage=np.where(ok["accession"].str.startswith("A0A"), "A0A* (recent)", "older"))
    print("\nby accession vintage:")
    print(ok.groupby("vintage")["has_model"].agg(n="size", coverage="mean").assign(
        coverage=lambda d: (d.coverage * 100).round(1)).to_string())

    hot, cool = ok[ok["ogt"] >= args.hot_cut]["has_model"], ok[ok["ogt"] < args.hot_cut]["has_model"]
    print(f"\nthermophile sample size n={len(hot)}, mesophile n={len(cool)}")
    if len(hot) < 10 or len(cool) < 10:
        print("  too few in one group to test; rerun with --stratify-ogt")
    if len(hot) >= 10 and len(cool) >= 10:
        from scipy import stats
        t = stats.fisher_exact([[hot.sum(), len(hot) - hot.sum()], [cool.sum(), len(cool) - cool.sum()]])
        print(f"\nthermophile (>=50C) coverage {hot.mean():.1%} vs mesophile {cool.mean():.1%}; "
              f"Fisher p = {t[1]:.3f}")
        print("  p < 0.05 would mean missing structures are biased by temperature -> AFDB alone is unsafe")


if __name__ == "__main__":
    main()
