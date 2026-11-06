#!/usr/bin/env python
"""Download prokaryotic S8 / S9 / S1 serine protease sequences from UniProt.

Output: data/processed/sequences.tsv with one row per protein:
accession, family, reviewed, name, taxid, organism, lineage, phylum, length, sequence,
act_sites, nuc_pos, nuc_source, pdb
"""
import csv
import io
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serprot.common import base_parser, find_nucleophile, load_config, parse_phylum, rpath, seed_all  # noqa: E402

SEARCH = "https://rest.uniprot.org/uniprotkb/search"
FIELDS = ["accession", "reviewed", "protein_name", "organism_id", "organism_name", "lineage",
          "length", "sequence", "ft_act_site", "xref_pdb"]
COLS = ["accession", "reviewed", "name", "taxid", "organism", "lineage", "length", "sequence", "act_raw", "pdb"]


def _parse_page(text: str) -> pd.DataFrame:
    """Parse one UniProt TSV page. UniProt never quotes fields, so quote handling is switched off."""
    df = pd.read_csv(io.StringIO(text), sep="\t", header=0, names=COLS, dtype=str, keep_default_na=False,
                     quoting=csv.QUOTE_NONE, on_bad_lines="skip")
    return df[df["length"].str.fullmatch(r"\d+")]


def _get(url, params, retries):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=300, headers={"User-Agent": "serprot-pipeline"})
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001
            wait = 10 * (attempt + 1)
            print(f"  request failed ({type(e).__name__}: {e}); retry {attempt + 1}/{retries} in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"UniProt request kept failing: {url}")


def fetch_family(pfam: str, taxa: list[int], lo: int, hi: int, retries: int = 6, page_size: int = 500) -> pd.DataFrame:
    """Cursor-paged download (documented UniProt method for large result sets). A single long /stream
    response was cut off at a fixed row count on this cluster, so we ask for small pages and follow the
    'next' links instead."""
    tax = " OR ".join(f"taxonomy_id:{t}" for t in taxa)
    query = f"(xref:pfam-{pfam}) AND ({tax}) AND (length:[{lo} TO {hi}])"
    url, params = SEARCH, {"query": query, "format": "tsv", "fields": ",".join(FIELDS), "size": page_size}
    frames, total, n_pages = [], -1, 0
    while url:
        r = _get(url, params, retries)
        if total < 0:
            total = int(r.headers.get("x-total-results", -1))
            print(f"  UniProt reports {total} entries for this query", flush=True)
        frames.append(_parse_page(r.text))
        n_pages += 1
        if n_pages % 20 == 0:
            print(f"  ... {sum(len(f) for f in frames)} rows after {n_pages} pages", flush=True)
        url, params = r.links.get("next", {}).get("url"), None
        time.sleep(0.1)
    df = pd.concat(frames, ignore_index=True).drop_duplicates("accession")
    print(f"  parsed {len(df)} rows in {n_pages} pages (UniProt reports {total})", flush=True)
    if total > 0 and len(df) < 0.98 * total:
        raise RuntimeError(f"{pfam}: got {len(df)} rows but UniProt reports {total}; download incomplete")
    return df


def main():
    ap = base_parser(__doc__)
    cfg = load_config(ap.parse_args().config)
    seed_all(cfg["seed"])
    lo, hi = cfg["length_range"]
    frames = []
    for fam, spec in cfg["families"].items():
        print(f"Fetching {fam} ({spec['pfam']}) ...", flush=True)
        df = fetch_family(spec["pfam"], cfg["taxa"], lo, hi)
        df["family"] = fam
        df["reviewed"] = df["reviewed"].eq("reviewed")
        print(f"  {len(df)} entries ({df['reviewed'].sum()} reviewed)")
        cap = cfg["max_per_family"]
        if len(df) > cap:
            rev = df[df["reviewed"]]
            rest = df[~df["reviewed"]].sample(n=max(cap - len(rev), 0), random_state=cfg["seed"])
            df = pd.concat([rev, rest])
            print(f"  subsampled to {len(df)}")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    # A protein can carry two Pfam domains (e.g. S8 + S9 in odd fusions): keep the first family seen.
    df = df.drop_duplicates("accession", keep="first").reset_index(drop=True)
    df["length"] = df["length"].astype(int)
    df["phylum"] = df["lineage"].map(parse_phylum)
    df["act_sites"] = df["act_raw"].map(lambda s: [int(x) for x in re.findall(r"ACT_SITE (\d+)", s)])
    nuc = [find_nucleophile(s, f, a) for s, f, a in zip(df["sequence"], df["family"], df["act_sites"])]
    df["nuc_pos"] = [n[0] for n in nuc]
    df["nuc_source"] = [n[1] for n in nuc]
    df["act_sites"] = df["act_sites"].map(lambda a: ",".join(map(str, a)))
    df = df.drop(columns=["act_raw"])
    df = df[df["sequence"].str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWYXBZUO]+")]  # drop odd entries
    print(df.groupby("family")["nuc_source"].value_counts().unstack(fill_value=0))

    out = rpath(cfg, "processed", "sequences.tsv")
    df.to_csv(out, sep="\t", index=False)
    print(f"Wrote {len(df)} sequences -> {out}")


if __name__ == "__main__":
    main()
