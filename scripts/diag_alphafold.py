#!/usr/bin/env python
"""Find out WHY AlphaFold requests fail here: real status codes, timings, and concurrency sensitivity.

Read-only, ~1 minute, makes well under 200 requests.
  python scripts/diag_alphafold.py --config config.yaml
"""
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serprot.common import base_parser, load_config, rpath  # noqa: E402

DIRECT = "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v{ver}.pdb"
API = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"
UA = {"User-Agent": "serprot-pipeline (academic research)"}


def probe(url, session=None, timeout=60):
    g = (session or requests).get
    t0 = time.time()
    try:
        r = g(url, timeout=timeout, headers=UA)
        body = r.content
        first_atom = body.find(b"ATOM ")
        return {"code": r.status_code, "secs": time.time() - t0, "bytes": len(body),
                "first_atom": first_atom, "retry_after": r.headers.get("Retry-After"),
                "ctype": r.headers.get("Content-Type", "")[:40]}
    except Exception as e:  # noqa: BLE001
        return {"code": f"EXC:{type(e).__name__}", "secs": time.time() - t0, "bytes": 0,
                "first_atom": -1, "retry_after": None, "ctype": ""}


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()
    cfg = load_config(args.config)
    accs = [a.strip() for a in Path(rpath(cfg, "processed", "structure_accessions.txt")).read_text().split() if a.strip()]
    sdir = Path(rpath(cfg, "structures", "x")).parent
    have = {f.name.split(".")[0] for f in sdir.glob("*.pdb.gz")}
    fresh = [a for a in accs if a not in have][: args.n]
    print(f"{len(accs)} targets, {len(have)} already downloaded; probing {len(fresh)} fresh ones\n")

    print("=== 1. ONE request at a time, no session reuse (the gentlest possible case) ===")
    c = Counter()
    for a in fresh[:8]:
        r = probe(DIRECT.format(acc=a, ver=4))
        c[str(r["code"])] += 1
        print(f"  {a}  v4 -> {str(r['code']):>18}  {r['secs']:5.2f}s  {r['bytes']:>9,}B  "
              f"first ATOM at {r['first_atom']:>7}  retry-after={r['retry_after']}  {r['ctype']}")
    print(f"  codes: {dict(c)}\n")

    print("=== 2. SHARED session, sequential (tests session reuse alone) ===")
    s = requests.Session()
    c = Counter()
    t0 = time.time()
    for a in fresh[8:16]:
        r = probe(DIRECT.format(acc=a, ver=4), session=s)
        c[str(r["code"])] += 1
    print(f"  codes: {dict(c)}   {(time.time() - t0) / 8:.2f}s per request\n")

    print("=== 3. CONCURRENCY sweep: shared session, N threads (this is what the real job does) ===")
    for nw in (1, 4, 8, 16):
        batch = (fresh * 4)[: nw * 3]
        sess = requests.Session()
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=nw) as ex:
            res = list(ex.map(lambda a: probe(DIRECT.format(acc=a, ver=4), session=sess), batch))
        cc = Counter(str(r["code"]) for r in res)
        el = time.time() - t0
        print(f"  {nw:>2} threads: {dict(cc)}  {len(batch) / el:5.2f} req/s  "
              f"median {sorted(r['secs'] for r in res)[len(res) // 2]:5.2f}s")

    print("\n=== 4. SEPARATE session per thread (does sharing one Session cause the failures?) ===")
    for nw in (8, 16):
        batch = (fresh * 4)[: nw * 3]
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=nw) as ex:
            res = list(ex.map(lambda a: probe(DIRECT.format(acc=a, ver=4), session=requests.Session()), batch))
        cc = Counter(str(r["code"]) for r in res)
        print(f"  {nw:>2} threads: {dict(cc)}  {len(batch) / (time.time() - t0):5.2f} req/s")

    print("\n=== 5. API endpoint on one accession ===")
    r = probe(API.format(acc=fresh[0]))
    print(f"  {fresh[0]} -> {r['code']}  {r['secs']:.2f}s  {r['bytes']}B  {r['ctype']}")

    print("\n=== 6. Is the first ATOM record ever past byte 8000? (my code only looked that far) ===")
    far = [(a, p["first_atom"], p["bytes"]) for a in fresh[:6]
           for p in [probe(DIRECT.format(acc=a, ver=4))] if p["first_atom"] > 0]
    if far:
        print(f"  first-ATOM offsets: {[f[1] for f in far]}   max file {max(f[2] for f in far):,}B")
        print(f"  past 8000: {sum(1 for f in far if f[1] > 8000)}/{len(far)}")
    else:
        print("  no successful downloads in this sample to measure")


if __name__ == "__main__":
    main()
