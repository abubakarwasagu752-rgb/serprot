#!/usr/bin/env python
"""Download AlphaFold DB models for the accessions chosen by 09_select_structures.py.

Strategy (changed after measuring: ~60% of our TrEMBL targets have no model, and EBI returns 429 under load):
  * ask the API FIRST -- one request settles a miss, two fetch a hit. Guessing file URLs cost three per miss.
  * a shared, self-tuning pacer slows every thread down when 429s appear and speeds back up when they stop,
    so the job finds the server's limit instead of fighting it.
  * 404 is permanent and recorded; 429/5xx/timeouts are transient and retried, and never land in no_model.txt.
Resumable and shardable. Models are stored gzipped as <accession>.pdb.gz.

  python scripts/10_fetch_structures.py --config config.yaml --workers 4
"""
import gzip
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serprot.common import base_parser, load_config, rpath  # noqa: E402

API = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"
UA = {"User-Agent": "serprot-pipeline (academic research; AlphaFold DB fetch)"}


class Pacer:
    """One shared, adaptive delay for all threads: widens on 429, narrows while requests succeed."""

    def __init__(self, start=0.0, ceiling=3.0):
        self.lock = threading.Lock()
        self.interval = start
        self.ceiling = ceiling
        self.next_at = 0.0
        self.n_429 = 0

    def wait(self):
        with self.lock:
            now = time.time()
            slot = max(now, self.next_at)
            self.next_at = slot + self.interval
        if slot > now:
            time.sleep(slot - now)

    def saw_429(self):
        with self.lock:
            self.n_429 += 1
            self.interval = min(max(self.interval * 1.6, 0.08), self.ceiling)

    def saw_ok(self):
        with self.lock:
            self.interval = max(self.interval * 0.98, 0.0)


def get(session, url, pacer, codes, timeout=60):
    pacer.wait()
    try:
        r = session.get(url, timeout=timeout, headers=UA)
    except Exception as e:  # noqa: BLE001
        codes[f"EXC:{type(e).__name__}"] += 1
        return None
    codes[str(r.status_code)] += 1
    if r.status_code == 429:
        pacer.saw_429()
        try:
            time.sleep(min(float(r.headers.get("Retry-After", 2)), 30))
        except (TypeError, ValueError):
            time.sleep(2)
        return None
    pacer.saw_ok()
    return r


def fetch_one(acc, sdir, session, pacer, codes, retries=4):
    """-> 'hit' | 'miss' (definitely no model) | 'error' (transient; safe to retry later)"""
    out = sdir / f"{acc}.pdb.gz"
    if out.exists() and out.stat().st_size > 0:
        return "hit"
    for _ in range(retries):
        r = get(session, API.format(acc=acc), pacer, codes)
        if r is None:
            continue
        if r.status_code == 404:
            return "miss"
        if r.status_code != 200:
            continue
        try:
            js = r.json()
        except Exception:  # noqa: BLE001
            continue
        if not js:
            return "miss"
        url = js[0].get("pdbUrl") or js[0].get("cifUrl")
        if not url:
            return "miss"
        c = get(session, url, pacer, codes, timeout=180)
        if c is None or c.status_code != 200 or b"ATOM " not in c.content:
            continue
        tmp = out.with_suffix(".tmp")
        tmp.write_bytes(gzip.compress(c.content))
        tmp.rename(out)
        return "hit"
    return "error"


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--workers", type=int, default=4,
                    help="4 is near EBI's comfortable limit here; 8+ triggered 429s in testing")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--retry-errors", action="store_true", help="also retry accessions logged as having no model")
    args = ap.parse_args()
    cfg = load_config(args.config)

    raw = Path(rpath(cfg, "processed", "structure_accessions.txt")).read_text().split()
    accs = list(dict.fromkeys(a.strip() for a in raw if a.strip()))
    sdir = Path(rpath(cfg, "structures", "x")).parent
    sdir.mkdir(parents=True, exist_ok=True)
    miss_file = sdir / "no_model.txt"
    known_miss = set() if args.retry_errors else set(miss_file.read_text().split()) if miss_file.exists() else set()
    todo = [a for i, a in enumerate(accs) if i % args.n_shards == args.shard and a not in known_miss]
    print(f"shard {args.shard}/{args.n_shards}: {len(todo)} to try, of {len(accs)} targets "
          f"({len(known_miss)} already known to have no model)", flush=True)

    counts, codes = Counter(), Counter()
    new_miss, lock, t0 = [], threading.Lock(), time.time()
    session, pacer = requests.Session(), Pacer()

    def work(acc):
        res = fetch_one(acc, sdir, session, pacer, codes)
        with lock:
            counts[res] += 1
            if res == "miss":
                new_miss.append(acc)
            n = sum(counts.values())
            if n % 200 == 0:
                rate = n / max(time.time() - t0, 1)
                print(f"  {n}/{len(todo)}  hit={counts['hit']} no-model={counts['miss']} err={counts['error']}  "
                      f"{rate:.1f}/s  pace={pacer.interval:.2f}s  429s={pacer.n_429}  "
                      f"eta {(len(todo) - n) / max(rate, 0.01) / 60:.0f} min", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(work, todo))

    if new_miss:
        with open(miss_file, "a") as fh:
            fh.write("\n".join(sorted(new_miss)) + "\n")
    have = len(list(sdir.glob("*.pdb.gz")))
    el = (time.time() - t0) / 60
    print(f"\ndone in {el:.1f} min: hit={counts['hit']} no-model={counts['miss']} transient-error={counts['error']}")
    print(f"HTTP status codes seen: {dict(codes)}")
    print(f"structures on disk: {have} ({100 * have / max(len(accs), 1):.1f}% of all targets) -> {sdir}")
    if counts["error"]:
        print(f"{counts['error']} transient failures; rerun the same command to pick them up (files already there are skipped)")


if __name__ == "__main__":
    main()
