#!/usr/bin/env python
"""Build data/raw/cleavages.tsv from the MEROPS SQL dump, and report how much usable specificity data exists.

Chain: Substrate_search (window of 8 residues, protease MER number)  ->  domain (MER number -> sequence_id)
       ->  alternative_id (sequence_id -> UniProt accession)  ->  keep proteases that are in our sequence set.
The SQL is parsed as text only. Synthetic substrates and theoretical cleavages are excluded by default.

  python scripts/merops_to_cleavages.py --merops-dir /shared/a2-public/abk/merops/extracted
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serprot.common import base_parser, cfg_file, load_config, rpath  # noqa: E402
from serprot.merops_sql import columns, iter_rows  # noqa: E402

AA3 = {"ala": "A", "arg": "R", "asn": "N", "asp": "D", "cys": "C", "gln": "Q", "glu": "E", "gly": "G", "his": "H", "ile": "I",
       "leu": "L", "lys": "K", "met": "M", "phe": "F", "pro": "P", "ser": "S", "thr": "T", "trp": "W", "tyr": "Y", "val": "V"}
SITES = ["Site_P4", "Site_P3", "Site_P2", "Site_P1", "Site_P1prime", "Site_P2prime", "Site_P3prime", "Site_P4prime"]
ACC = re.compile(r"^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})$")


def one(x):
    return AA3.get((x or "").strip().lower(), "-")


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--merops-dir", required=True)
    ap.add_argument("--types", default="physiological,non-physiological,pathological")
    ap.add_argument("--min-report", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    d, types = Path(a.merops_dir), set(a.types.split(","))

    f = str(d / "Substrate_search.sql")
    cols = columns(f, "Substrate_search"); ix = {c: i for i, c in enumerate(cols)}
    rows, seen = [], set()
    for r in iter_rows(f, "Substrate_search"):
        mer, ctype = r[ix["mernum"]], r[ix["cleavage_type"]]
        if not mer or ctype not in types:
            continue
        win = "".join(one(r[ix[c]]) for c in SITES)
        if win[3] == "-" or 8 - win.count("-") < 4:
            continue
        key = (mer, r[ix["Uniprot"]] or r[ix["Substrate_name"]], r[ix["resnum"]], win)
        if key in seen:
            continue
        seen.add(key)
        rows.append((mer, r[ix["code"]], ctype, win, len(rows)))   # last item: unique cleavage-event id
    w = pd.DataFrame(rows, columns=["mernum", "code", "cleavage_type", "window", "event"])
    print(f"windows kept: {len(w)} for {w['mernum'].nunique()} MEROPS protease sequences (types: {sorted(types)})")

    need = set(w["mernum"])
    dm = str(d / "domain.sql"); dc = columns(dm, "domain"); di = {c: i for i, c in enumerate(dc)}
    mer2seq = {}
    for r in iter_rows(dm, "domain"):
        if r[di["mernum"]] in need:
            mer2seq[r[di["mernum"]]] = r[di["sequence_id"]]
    seqids = set(mer2seq.values())
    af = str(d / "alternative_id.sql"); ac = columns(af, "alternative_id"); ai = {c: i for i, c in enumerate(ac)}
    seq2acc = defaultdict(set)
    for r in iter_rows(af, "alternative_id"):
        if r[ai["type"]] == "uniprot" and r[ai["sequence_id"]] in seqids:
            seq2acc[r[ai["sequence_id"]]].add(r[ai["alternative_id"]])
    for s in seqids:                      # sequence_id that is itself an accession
        if not seq2acc[s] and ACC.match(s):
            seq2acc[s].add(s)
    print(f"MER numbers found in domain table: {len(mer2seq)}/{len(need)}; with a UniProt accession: "
          f"{sum(1 for m, s in mer2seq.items() if seq2acc[s])}")

    # one row per (cleavage event, UniProt accession); events are already de-duplicated above, and two different
    # events with identical residues must both be kept
    m = pd.DataFrame([(acc, mer, c, t, wi, ev) for mer, c, t, wi, ev in rows for acc in seq2acc.get(mer2seq.get(mer), [])],
                     columns=["protease_acc", "mernum", "code", "cleavage_type", "window", "event"]
                     ).drop_duplicates(["protease_acc", "event"]).drop(columns="event")
    all_out = rpath(cfg, "raw", "merops_cleavages_all_mapped.tsv")
    m.to_csv(all_out, sep="\t", index=False)
    print(f"mapped to UniProt: {len(m)} windows, {m['protease_acc'].nunique()} proteases (any organism) -> {all_out}")

    seqs = pd.read_csv(rpath(cfg, "processed", "sequences.tsv"), sep="\t", usecols=["accession", "family", "organism"])
    lab = pd.read_csv(rpath(cfg, "processed", "labels.tsv"), sep="\t", usecols=["accession", "ogt"])
    ours = m[m["protease_acc"].isin(set(seqs["accession"]))]
    out = Path(a.out) if a.out else cfg_file(cfg, cfg["specificity"]["cleavages"])
    out.parent.mkdir(parents=True, exist_ok=True)
    ours.to_csv(out, sep="\t", index=False)
    print(f"\nproteases that are in OUR prokaryotic sequence set: {ours['protease_acc'].nunique()} "
          f"({len(ours)} windows) -> {out}")

    cnt = ours.groupby("protease_acc").size().rename("n_cleavages").reset_index()
    cnt = cnt.merge(seqs, left_on="protease_acc", right_on="accession").merge(lab, on="accession", how="left")
    cnt["code"] = cnt["protease_acc"].map(ours.groupby("protease_acc")["code"].first())
    print("\nper family: proteases with >= N windows  |  of those, with a growth temperature  |  with OGT >= 50 C")
    for fam, g in cnt.groupby("family"):
        line = [fam]
        for n in (1, 10, 20, 50):
            s = g[g["n_cleavages"] >= n]
            line.append(f">={n}: {len(s)} | ogt {int(s['ogt'].notna().sum())} | hot {int((s['ogt'] >= 50).sum())}")
        print("  " + "   ".join(line))
    show = cnt[cnt["n_cleavages"] >= a.min_report].sort_values(["family", "n_cleavages"], ascending=[True, False])
    print(f"\nprokaryotic proteases with >= {a.min_report} windows ({len(show)}):")
    print(show[["family", "code", "protease_acc", "n_cleavages", "ogt", "organism"]].to_string(index=False))


if __name__ == "__main__":
    main()
