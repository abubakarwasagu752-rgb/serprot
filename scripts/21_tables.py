#!/usr/bin/env python
"""Manuscript tables and supplementary tables, built from the analysis tables stages 12-19 wrote.

Main tables are rendered as pipe-table markdown you can paste into the manuscript and as CSV.
Supplementary goes to one CSV per table plus a single multi-sheet .xlsx when openpyxl is available.

  Table 1   pocket vs matched control, the descriptors surviving FDR, in their own units
  Table 2   what actually entered the analysis, by family and by domain
  S1-S8     the full grids behind every claim: all descriptors, all radii, both structure sets,
            the distance profile, the burial grid, attribution, flexibility and the domain split

Counts in Table 2 are computed from the proteins that passed every filter and entered the
regressions, not from what was downloaded -- those differ and only the first is reportable.

  python scripts/21_tables.py --config config.yaml
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serprot.common import base_parser, load_config, load_dataset, rpath  # noqa: E402

UNITS = {"resvol": "Å³", "sasa_mean": "Å²", "sasa_total": "Å²",
         "rgyr": "Å", "hydropathy": "KD units", "net_charge": "e", "n": "residues",
         "plddt": "pLDDT", "burial": "neighbours"}
PRETTY = {"resvol": "Mean residue volume", "f_polar": "Polar fraction", "f_pro": "Proline fraction",
          "f_ivywrel": "IVYWREL fraction", "f_neg": "Acidic fraction", "f_pos": "Basic fraction",
          "f_aromatic": "Aromatic fraction", "f_small": "Small-residue fraction",
          "f_gly": "Glycine fraction", "hydropathy": "Mean hydropathy", "net_charge": "Net charge",
          "sasa_mean": "Mean SASA", "sasa_total": "Total SASA", "burial": "Burial",
          "rgyr": "Radius of gyration", "plddt": "pLDDT", "n": "Patch size",
          "attribution": "ESM-2 attribution"}
FAMILY_PFAM = {"S8": "PF00082", "S9": "PF00326", "S1": "PF00089"}


MINUS = "−"   # a real minus sign, not a hyphen; these tables are read, not parsed


def sig(v, dec=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:+.{dec}f}".replace("-", MINUS)


def ci(v, lo, hi, dec=3):
    return f"{sig(v, dec)} [{sig(lo, dec)}, {sig(hi, dec)}]"


def scale_of(k):
    """Composition fractions are ~1e-3, so they are reported in units of 1e-3 rather than as a
    column of leading zeros. Anything with a real unit keeps it."""
    return (1.0, UNITS[k]) if k in UNITS else (1e3, "×10⁻³")


def qfmt(q):
    if not np.isfinite(q):
        return "—"
    return "< 0.001" if q < 0.001 else f"{q:.3f}"


def md_table(df, align=None):
    cols = list(df.columns)
    a = align or (["---"] * len(cols))
    out = ["| " + " | ".join(cols) + " |", "| " + " | ".join(a) + " |"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


class Tables:
    def __init__(self, cfg):
        self.dir = Path(rpath(cfg, "runs", "analysis", "x.tsv")).parent

    def get(self, name, required=True):
        p = self.dir / name
        if p.exists():
            return pd.read_csv(p, sep="\t")
        if required:
            raise FileNotFoundError(name)
        print(f"  (absent: {name})")
        return None


def analysed_set(cfg, tag, radius, min_plddt):
    """The proteins that survived every filter and entered the regressions, with their domain."""
    rx = re.compile(rf"^pocket_descriptors{re.escape(tag)}(\.\d+)?\.tsv$")
    base = Path(rpath(cfg, "processed", "x.tsv")).parent
    parts = [p for p in sorted(base.glob("pocket_descriptors*.tsv")) if rx.match(p.name)]
    if not parts:
        return None
    d = pd.concat([pd.read_csv(p, sep="\t") for p in parts], ignore_index=True)
    d = d.drop_duplicates("accession")
    if "status" in d:
        d = d[d["status"] == "ok"]
    col = f"pk{radius}_plddt"
    if col in d:
        d = d[d[col] >= min_plddt]
    tg = pd.read_csv(rpath(cfg, "processed", "structure_targets.tsv"), sep="\t")
    d = d[["accession"]].merge(tg[["accession", "family", "cluster_pairs", "ogt"]],
                               on="accession", how="inner")
    good = d.groupby("cluster_pairs")["ogt"].transform("size") >= 2
    spread = d.groupby("cluster_pairs")["ogt"].transform(lambda s: s.max() - s.min()) > 0
    d = d[good & spread].copy()
    ds = load_dataset(cfg)
    if "lineage" in ds:
        lin = ds.set_index("accession")["lineage"]
        d["domain"] = d["accession"].map(lin).map(
            lambda s: (re.search(r"\b(Archaea|Bacteria|Eukaryota)\b", s).group(1)
                       if isinstance(s, str) and re.search(r"\b(Archaea|Bacteria|Eukaryota)\b", s)
                       else "Unknown"))
    else:
        d["domain"] = "Unknown"
    return d


def composition(d, label):
    rows = []
    for key, grp in [(f, d[d["family"] == f]) for f in ["S8", "S9", "S1"] if (d["family"] == f).any()] + \
                    [(dm, d[d["domain"] == dm]) for dm in ["Bacteria", "Archaea"]
                     if (d["domain"] == dm).any()]:
        hot = grp[grp["ogt"] >= 50]
        rows.append({"Structures": label, "Group": key, "Pfam": FAMILY_PFAM.get(key, "—"),
                     "Proteins": f"{len(grp):,}", "Clades": f"{grp['cluster_pairs'].nunique():,}",
                     "Clades with a thermophile": f"{hot['cluster_pairs'].nunique():,}",
                     "Median OGT (°C)": f"{grp['ogt'].median():.0f}",
                     "OGT range (°C)": f"{grp['ogt'].min():.0f}–{grp['ogt'].max():.0f}"})
    return pd.DataFrame(rows)


def table1(T, radius):
    d = T.get(f"pocket_vs_temperature_r{radius}.tsv")
    raw = d[(d["family"] == "all") & (d["adjustment"] == "raw")]
    std = d[(d["family"] == "all") & (d["adjustment"] == "standardized")].set_index("descriptor")
    keep = raw[(raw["q_fdr"] < 0.05) & (raw["descriptor"] != "n")].sort_values("q_fdr")
    rows = []
    for _, r in keep.iterrows():
        k = r["descriptor"]
        m, unit = scale_of(k)
        rows.append({"Descriptor": f"{PRETTY.get(k, k)} ({unit})",
                     "Pocket": sig(r["slope_pocket"] * m), "Control": sig(r["slope_control"] * m),
                     "Difference [95% CI]": ci(r["difference"] * m, r["difference_lo"] * m,
                                               r["difference_hi"] * m),
                     "Difference (SD)": sig(std["difference"].get(k, np.nan)),
                     "q": qfmt(r["q_fdr"])})
    return pd.DataFrame(rows), int(keep["n"].max()), int(keep["clades"].max())


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--radius", type=int, default=10)
    ap.add_argument("--min-plddt", type=float, default=70.0)
    ap.add_argument("--outdir", default=None, help="default: <root>/tables")
    args = ap.parse_args()
    cfg = load_config(args.config)
    T = Tables(cfg)
    # config.yaml has no tables entry, so take it off the project root rather than make the user
    # edit their config just to run this
    outdir = (Path(args.outdir).expanduser() if args.outdir
              else Path(cfg["paths"]["root"]).expanduser() / cfg["paths"].get("tables", "tables"))
    outdir.mkdir(parents=True, exist_ok=True)
    md, sheets = [], {}

    print("main tables")
    t1, n1, k1 = table1(T, args.radius)
    t1.to_csv(outdir / "table1.csv", index=False)
    md.append(f"**Table 1.** Within-clade change per +10 °C of optimal growth temperature in the "
              f"catalytic neighbourhood ({args.radius} Å) and in a size- and burial-matched "
              f"control patch on the same protein, for the descriptors surviving Benjamini-Hochberg "
              f"correction. Slopes are in each descriptor's own units; the SD column restates the "
              f"difference in units of that descriptor's within-clade standard deviation. "
              f"n = {n1:,} proteins in {k1:,} clades, AlphaFold models.\n\n" + md_table(t1))

    comps = []
    for tag, label in [("", "AlphaFold"), ("_esmfold", "ESMFold")]:
        d = analysed_set(cfg, tag, args.radius, args.min_plddt)
        if d is not None:
            comps.append(composition(d, label))
    if comps:
        t2 = pd.concat(comps, ignore_index=True)
        t2.to_csv(outdir / "table2.csv", index=False)
        md.append("**Table 2.** Proteins entering the analysis, by family and by domain. Counts are "
                  "after structure, confidence and clade filters, so they are smaller than the "
                  "number of sequences retrieved. A clade is an MMseqs2 cluster at 50% identity; "
                  "thermophile means an optimal growth temperature of at least 50 °C.\n\n"
                  + md_table(t2))
        sheets["S0_composition"] = t2

    print("supplementary tables")
    supp = [("S1_pocket_vs_control_alphafold", f"pocket_vs_temperature_r{args.radius}.tsv", True),
            ("S2_pocket_vs_control_esmfold", f"pocket_vs_temperature_esmfold_r{args.radius}.tsv", False),
            ("S3_radius_6A", "pocket_vs_temperature_r6.tsv", False),
            ("S4_radius_8A", "pocket_vs_temperature_r8.tsv", False),
            ("S5_distance_profile", "distance_profile.tsv", False),
            ("S6_burial_by_distance", "burial_distance.tsv", False),
            ("S7_residue_attribution", f"residue_attribution_r{args.radius}.tsv", False),
            ("S8_flexibility_alphafold", "flexibility.tsv", False),
            ("S9_flexibility_esmfold", "flexibility_esmfold.tsv", False),
            ("S10_flexibility_profile", "flexibility_profile.tsv", False),
            ("S11_subgroups_domain", "subgroups_domain.tsv", False)]
    for name, fn, required in supp:
        d = T.get(fn, required=required)
        if d is None:
            continue
        d.to_csv(outdir / f"{name}.csv", index=False)
        sheets[name] = d
        print(f"  {name}: {len(d)} rows")

    (outdir / "tables.md").write_text("\n\n".join(md) + "\n")
    try:
        with pd.ExcelWriter(outdir / "supplementary_tables.xlsx", engine="openpyxl") as xl:
            for name, d in sheets.items():
                d.to_excel(xl, sheet_name=name[:31], index=False)
        print(f"\nwrote supplementary_tables.xlsx ({len(sheets)} sheets)")
    except ImportError:
        print("\nno openpyxl, so the CSVs are the supplementary output:")
        print("  conda activate serprot && pip install openpyxl")

    print(f"\nall output in {outdir}")
    for p in sorted(outdir.iterdir()):
        print(f"  {p.name}")
    print("\nTable 1 and Table 2 are in tables.md as pipe tables, ready to paste.")


if __name__ == "__main__":
    main()
