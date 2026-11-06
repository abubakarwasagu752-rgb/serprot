#!/usr/bin/env python
"""Read-only look at the MEROPS SQL dump: how much cleavage data exists for S1/S8/S9, and how proteases map to sequences.

  python scripts/inspect_merops.py --dir /shared/a2-public/abk/merops/extracted > merops_inspect.txt
"""
import argparse
import itertools
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serprot.merops_sql import columns, iter_rows  # noqa: E402

FAMS = ("S01", "S08", "S09")


def peek(d, table, n=3, width=60):
    f = d / f"{table}.sql"
    if not f.exists():
        print(f"\n--- {table}: file not found"); return
    cols = columns(str(f), table)
    print(f"\n--- {table}  columns: {cols}")
    for r in itertools.islice(iter_rows(str(f), table), n):
        print("   ", {c: (v[:width] if isinstance(v, str) else v) for c, v in zip(cols, r)})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True)
    d = Path(ap.parse_args().dir)

    f = str(d / "Substrate_search.sql"); cols = columns(f, "Substrate_search"); ix = {c: i for i, c in enumerate(cols)}
    n, types, per = 0, Counter(), defaultdict(Counter)
    fam_rows, fam_codes, fam_mer, fam_prot = Counter(), defaultdict(set), Counter(), defaultdict(Counter)
    for r in iter_rows(f, "Substrate_search"):
        n += 1
        code, ctype = r[ix["code"]] or "", r[ix["cleavage_type"]]
        types[ctype] += 1
        fam = code[:3]
        if fam in FAMS:
            fam_rows[fam] += 1; fam_codes[fam].add(code)
            if r[ix["mernum"]]: fam_mer[fam] += 1
            fam_prot[fam][(code, r[ix["Protease"]], r[ix["organism"]], r[ix["mernum"]])] += 1
    print(f"Substrate_search: {n} rows. cleavage_type counts: {dict(types)}")
    for fam in FAMS:
        print(f"\n== {fam}: {fam_rows[fam]} rows, {len(fam_codes[fam])} distinct peptidase codes, {fam_mer[fam]} rows carry a mernum")
        big = [(k, v) for k, v in fam_prot[fam].most_common(400) if v >= 20]
        print(f"   (code, protease name, organism, mernum) with >= 20 cleavages: {len(big)}  -> top 25:")
        for k, v in big[:25]:
            print(f"   {v:6d}  {k}")

    c = str(d / "cleavage.sql"); cc = columns(c, "cleavage"); ci = {k: i for i, k in enumerate(cc)}
    cr, cm = Counter(), defaultdict(set)
    for r in iter_rows(c, "cleavage"):
        fam = (r[ci["code"]] or "")[:3]
        if fam in FAMS:
            cr[fam] += 1
            if r[ci["mernum"]]: cm[fam].add(r[ci["mernum"]])
    print("\ncleavage table rows for S01/S08/S09:", dict(cr), "| distinct mernum:", {k: len(v) for k, v in cm.items()})

    for t in ["alternative_id", "domain", "code_summary", "organism_name", "family", "sequence"]:
        if t == "sequence":   # 3 GB, mostly sequence text: columns only
            print(f"\n--- sequence columns: {columns(str(d / 'sequence.sql'), 'sequence')} (not read, 3 GB)"); continue
        peek(d, t)


if __name__ == "__main__":
    main()
