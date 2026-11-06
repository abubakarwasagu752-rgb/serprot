#!/usr/bin/env python
"""Convert a TEMPURA export (CSV/TSV from togodb.org/db/tempura) into data/raw/ogt_table.csv.

Output columns: taxid, organism, ogt.  Column names are matched loosely (case-insensitive), because exports vary.
OGT = Topt when present, otherwise the midpoint of Topt_low / Topt_high.

  python scripts/prep_tempura.py --input /path/to/tempura_export.csv --out data/raw/ogt_table.csv
"""
import argparse
import re
from pathlib import Path

import pandas as pd


def find(cols, *patterns, exclude=()):
    for pat in patterns:
        for c in cols:
            n = c.strip().lower()
            if re.search(pat, n) and not any(re.search(e, n) for e in exclude):
                return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="data/raw/ogt_table.csv")
    a = ap.parse_args()
    df = pd.read_csv(a.input, sep=None, engine="python")
    cols = list(df.columns)
    print("columns found:", cols)
    c_tax = find(cols, r"^tax(on)?(omy)?[_ ]?id$", r"tax.*id", r"^taxid$")
    c_org = find(cols, r"genus_and_species", r"^species", r"organism", r"species", r"^name$", r"strain")
    c_top = find(cols, r"^topt$", r"^ogt$", r"topt(?!.*(low|high))", r"optim", exclude=(r"low", r"high"))
    c_lo, c_hi = find(cols, r"topt.*low", r"optim.*low"), find(cols, r"topt.*high", r"optim.*high")
    print(f"using: taxid={c_tax!r}, organism={c_org!r}, topt={c_top!r}, topt_low={c_lo!r}, topt_high={c_hi!r}")
    if c_org is None or (c_top is None and not (c_lo and c_hi)):
        raise SystemExit("could not identify organism / optimum-temperature columns; check the printed column names")
    ogt = pd.to_numeric(df[c_top], errors="coerce") if c_top else pd.Series(float("nan"), index=df.index)
    if c_lo and c_hi:
        mid = (pd.to_numeric(df[c_lo], errors="coerce") + pd.to_numeric(df[c_hi], errors="coerce")) / 2
        ogt = ogt.fillna(mid)
    out = pd.DataFrame({"taxid": pd.to_numeric(df[c_tax], errors="coerce").astype("Int64") if c_tax else pd.NA,
                        "organism": df[c_org].astype(str).str.strip().str.strip('"').str.strip(), "ogt": ogt}).dropna(subset=["ogt"])
    out = out[(out.ogt > -5) & (out.ogt < 125)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"wrote {len(out)} rows -> {a.out}  (OGT range {out.ogt.min():.0f}-{out.ogt.max():.0f} C, "
          f"{(out.ogt >= 50).sum()} thermophilic rows >= 50 C)")


if __name__ == "__main__":
    main()
