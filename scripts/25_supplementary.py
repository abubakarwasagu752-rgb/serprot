#!/usr/bin/env python
"""Build the single supplementary file the journal requires.

Bioinformatics asks that all additional text, figures and tables be presented in ONE file, submitted
separately from the manuscript, referred to from the main text, and never altered after acceptance.
This writes Supplementary_Material.pdf: a title page, a contents list, a legend for every table and
the tables themselves, paginated.

It also writes the same tables to one .xlsx, which is not the submission file but is far more useful
to anyone re-checking the numbers, and belongs in the public repository.

Two tables that earlier runs omitted are included here, and one of them matters: the ESMFold burial
grid is cited in the Results and had no supplementary table behind it.

  python scripts/25_supplementary.py --config config.yaml
"""
import sys
import textwrap
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serprot.common import base_parser, load_config, rpath  # noqa: E402

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
except ImportError:
    raise SystemExit("matplotlib is missing:\n  conda activate serprot && pip install matplotlib")

TITLE = "Thermal adaptation spares the catalytic neighbourhood in bacterial serine proteases"

# (number, file, short title, legend, columns to show)
SPEC = [
    ("S1", "pocket_vs_temperature_r10.tsv", "Pocket versus matched control, AlphaFold models",
     "Within-clade change in each descriptor per +10 °C of optimal growth temperature, in the "
     "catalytic neighbourhood (10 Å) and in the size- and burial-matched control patch on the same "
     "protein. Three passes are reported: raw descriptor values, values adjusted for local pLDDT, and "
     "values standardised by each descriptor's within-clade standard deviation. Intervals are 95% "
     "cluster bootstrap over clades; p from within-clade permutation; q by Benjamini-Hochberg.", None),
    ("S2", "pocket_vs_temperature_esmfold_r10.tsv", "Pocket versus matched control, ESMFold models",
     "As Supplementary Table S1, computed on structures predicted with ESMFold for sequences that "
     "AlphaFold DB does not cover. The two structure sets barely overlap.", None),
    ("S3", "pocket_vs_temperature_r6.tsv", "Pocket versus matched control at a 6 Å radius",
     "As Supplementary Table S1 with the catalytic neighbourhood defined by a 6 Å radius, testing "
     "whether the choice of radius carries the result.", None),
    ("S4", "pocket_vs_temperature_r8.tsv", "Pocket versus matched control at an 8 Å radius",
     "As Supplementary Table S1 with an 8 Å radius.", None),
    ("S5", "distance_profile.tsv", "Descriptor and attribution slopes by distance shell",
     "Within-clade slope on growth temperature for each measure within concentric shells around the "
     "catalytic serine. The burial-adjusted pass partials local burial out within each shell; as "
     "noted in the Discussion this cannot address a between-shell confound, which is what "
     "Supplementary Tables S6 and S7 address instead.", None),
    ("S6", "burial_distance.tsv", "Distance shells stratified by solvent exposure, AlphaFold models",
     "Slopes recomputed inside terciles of solvent exposure taken within each protein, so that every "
     "comparison is between residues of comparable exposure. Stratum 0 is the most exposed tercile.",
     None),
    ("S7", "burial_distance_esmfold.tsv",
     "Distance shells stratified by solvent exposure, ESMFold models",
     "As Supplementary Table S6, on the ESMFold structure set. The inner-shell IVYWREL p-values "
     "quoted in the Results come from this table.", None),
    ("S8", "residue_attribution_r10.tsv", "Exact per-residue attribution by region",
     "Within-clade slope of each region's ESM-2 attribution readout on growth temperature, per "
     "+10 °C. The decomposition is exact: per-residue contributions sum to the model's own "
     "prediction. The control-minus-pocket row carries the bootstrap interval and permutation p.",
     None),
    ("S9", "flexibility.tsv", "Elastic-network flexibility, AlphaFold models",
     "Within-clade slope of relative Cα fluctuation on growth temperature, pocket against matched "
     "control, by family. Fluctuations come from a Gaussian Network Model with a 7.3 Å cutoff, "
     "normalised by each protein's own mean; the unnormalised pass is reported alongside.", None),
    ("S10", "flexibility_esmfold.tsv", "Elastic-network flexibility, ESMFold models",
     "As Supplementary Table S9, on the ESMFold structure set.", None),
    ("S11", "flexibility_profile.tsv", "Flexibility by distance shell",
     "Mean relative fluctuation and its within-clade slope on growth temperature, by distance shell. "
     "The profile is strongly structured while the slopes are null throughout.", None),
    ("S12", "subgroups_domain.tsv", "Pocket versus control split by domain",
     "The analysis of Supplementary Table S1 repeated within bacteria and within archaea, with an "
     "interaction test permuting clade-to-domain assignments. Rows labelled Bacteria_vs_Archaea "
     "carry the difference of differences and its permutation p.", None),
    ("S13", "heldout_eval.tsv", "Independent test-set performance",
     "Performance of the ESM-2 ridge regressor on a partition of 30%-identity clusters used in "
     "neither model fitting nor penalty selection, against amino-acid composition and a "
     "training-mean baseline under the identical protocol. Reported as the journal requires in "
     "place of cross-validation averages.", None),
    ("S14", "heldout_composition.tsv", "Composition of the training and held-out partitions",
     "Size, cluster count, family breakdown and growth-temperature distribution of the two "
     "partitions, showing they are closely matched and share no sequence cluster.", None),
]


def fmt(v):
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, float) or isinstance(v, np.floating):
        if not np.isfinite(v):
            return "-"
        if v == 0:
            return "0"
        return f"{v:.4g}" if abs(v) >= 1e-4 else f"{v:.2e}"
    return str(v)


def as_text(df: pd.DataFrame, cols=None):
    d = df[cols] if cols else df
    body = d.map(fmt).astype(str)
    widths = {c: max(len(str(c)), *(len(x) for x in body[c])) if len(body) else len(str(c))
              for c in body.columns}
    head = "  ".join(str(c).rjust(widths[c]) for c in body.columns)
    rows = ["  ".join(body.iloc[i][c].rjust(widths[c]) for c in body.columns)
            for i in range(len(body))]
    return head, rows


def page(pdf, lines, fontsize, landscape=True):
    w, h = (11.69, 8.27) if landscape else (8.27, 11.69)
    fig = plt.figure(figsize=(w, h))
    fig.text(0.045, 0.955, "\n".join(lines), family="monospace", fontsize=fontsize,
             va="top", ha="left", linespacing=1.45)
    pdf.savefig(fig)
    plt.close(fig)


def write_xlsx(outdir: Path, loaded):
    xlsx = outdir / "Supplementary_Tables.xlsx"
    try:
        with pd.ExcelWriter(xlsx, engine="openpyxl") as xl:
            for num, _short, _lg, cols, d in loaded:
                (d[cols] if cols else d).to_excel(xl, sheet_name=f"{num}", index=False)
        print(f"wrote {xlsx}")
    except ImportError:
        print("no openpyxl, so no spreadsheet was written:  pip install openpyxl")


def _oxml(tag, **attrs):
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    el = OxmlElement(tag)
    for k, v in attrs.items():
        el.set(qn(f"w:{k}"), v)
    return el


def _tight_cells(tbl, cm=0.04):
    """python-docx leaves 0.19 cm of padding either side of every cell. On a twenty-column table
    that is a quarter of the page spent on whitespace, which is what forces numbers to break
    across lines."""
    mar = _oxml("w:tblCellMar")
    for side in ("top", "left", "bottom", "right"):
        mar.append(_oxml(f"w:{side}", w=str(int(cm * 567)), type="dxa"))
    tbl._tbl.tblPr.append(mar)


def _fixed_layout(tbl):
    """Honour the column widths we compute instead of letting Word re-fit the table."""
    tbl._tbl.tblPr.append(_oxml("w:tblLayout", type="fixed"))


def _repeat_header(row):
    """Reprint the header row on every page: a 224-row table otherwise runs for eight pages with
    no indication of which column is which."""
    row._tr.get_or_add_trPr().append(_oxml("w:tblHeader", val="true"))


def _col_widths(dd, usable_cm):
    """Widths proportional to the longest value in each column, so no number has to wrap. Headers
    may wrap, so each claims only the width of its longest single word."""
    body = dd.map(fmt).astype(str)
    need = []
    for c in dd.columns:
        word = max((len(w) for w in str(c).replace("_", " ").split()), default=1)
        val = max((len(x) for x in body[c]), default=1) if len(body) else 1
        need.append(max(val, word, 3))
    total = float(sum(need))
    return [usable_cm * n / total for n in need]


def build_docx(path: Path, loaded, author: str):
    """Word version: real tables rather than monospace text, which editors can reflow and copy from."""
    from docx import Document
    from docx.enum.section import WD_ORIENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Pt, Cm

    doc = Document()
    sec = doc.sections[0]
    sec.orientation, sec.page_width, sec.page_height = WD_ORIENT.LANDSCAPE, Cm(29.7), Cm(21.0)
    for m in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, m, Cm(1.5))
    normal = doc.styles["Normal"]
    normal.font.name, normal.font.size = "Arial", Pt(9)

    h = doc.add_paragraph()
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = h.add_run("SUPPLEMENTARY MATERIAL")
    r.bold, r.font.size = True, Pt(14)
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t.add_run(TITLE).italic = True
    a = doc.add_paragraph()
    a.alignment = WD_ALIGN_PARAGRAPH.CENTER
    a.add_run(f"{author}\nCompiled {date.today().isoformat()}")

    doc.add_paragraph()
    doc.add_paragraph().add_run("Contents").bold = True
    for num, short, _lg, _c, d in loaded:
        doc.add_paragraph(f"Supplementary Table {num}. {short} ({len(d)} rows)",
                          style="List Bullet")
    doc.add_page_break()

    for num, short, legend, cols, d in loaded:
        cap = doc.add_paragraph()
        cr = cap.add_run(f"Supplementary Table {num}. {short}")
        cr.bold, cr.font.size = True, Pt(10)
        lg = doc.add_paragraph(legend)
        lg.runs[0].font.size = Pt(8.5)

        dd = (d[cols] if cols else d)
        size = Pt(7) if len(dd.columns) <= 13 else Pt(6)
        widths = _col_widths(dd, 29.7 - 3.0)

        tbl = doc.add_table(rows=1, cols=len(dd.columns))
        tbl.style = "Table Grid"
        tbl.autofit = False
        _fixed_layout(tbl)
        _tight_cells(tbl)
        _repeat_header(tbl.rows[0])

        def put(cell, text, width, bold=False):
            cell.width = Cm(width)
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            run = p.add_run(text)
            run.bold, run.font.size = bold, size

        for j, c in enumerate(dd.columns):
            put(tbl.rows[0].cells[j], str(c).replace("_", " "), widths[j], bold=True)
        for _, row in dd.iterrows():
            cells = tbl.add_row().cells
            for j, c in enumerate(dd.columns):
                put(cells[j], fmt(row[c]), widths[j])
        doc.add_page_break()

    doc.save(path)


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--author", default="Wasagu, A.")
    ap.add_argument("--rows-per-page", type=int, default=42)
    ap.add_argument("--format", default="docx,pdf,xlsx",
                    help="which outputs to write; docx is the editable submission file")
    args = ap.parse_args()
    want = set(args.format.split(","))
    cfg = load_config(args.config)
    adir = Path(rpath(cfg, "runs", "analysis", "x.tsv")).parent
    outdir = Path(cfg["paths"]["root"]).expanduser() / cfg["paths"].get("tables", "tables")
    outdir.mkdir(parents=True, exist_ok=True)

    loaded, missing = [], []
    for num, fn, short, legend, cols in SPEC:
        p = adir / fn
        if p.exists():
            loaded.append((num, short, legend, cols, pd.read_csv(p, sep="\t")))
        else:
            missing.append((num, fn))
    for num, fn in missing:
        print(f"  MISSING {num}: {fn}")

    if "docx" in want:
        docx_path = outdir / "Supplementary_Material.docx"
        try:
            build_docx(docx_path, loaded, args.author)
            print(f"wrote {docx_path}  ({docx_path.stat().st_size / 1024:.0f} KB)")
        except ImportError:
            print("python-docx is missing, so no Word file was written:")
            print("  conda activate serprot && pip install python-docx")

    if "pdf" not in want:
        if "xlsx" in want:
            write_xlsx(outdir, loaded)
        return

    pdf_path = outdir / "Supplementary_Material.pdf"
    with PdfPages(pdf_path) as pdf:
        front = ["", "SUPPLEMENTARY MATERIAL", "", *textwrap.wrap(TITLE, 86), "",
                 args.author, "", f"Compiled {date.today().isoformat()}", "", "",
                 "CONTENTS", ""]
        for num, short, _lg, _c, d in loaded:
            front.append(f"  Supplementary Table {num:<4} {short}")
            front.append(f"  {'':<24} {len(d)} rows")
        front += ["", "",
                  *textwrap.wrap("All tables are generated directly from the analysis pipeline and "
                                 "are also available as a spreadsheet and as tab-separated files in "
                                 "the public repository, together with the code that produced them.",
                                 86)]
        page(pdf, front, 10, landscape=False)

        for num, short, legend, cols, d in loaded:
            head, rows = as_text(d, cols)
            # pick the largest readable size at which the widest line still fits the page
            need = max(len(head), *(len(r) for r in rows)) if rows else len(head)
            size = next((s for s in (8, 7.5, 7, 6.5, 6, 5.5, 5) if need <= (10.7 * 72) / (0.6 * s)), 5)
            lines = [f"Supplementary Table {num}. {short}", ""]
            lines += textwrap.wrap(legend, int((10.7 * 72) / (0.6 * size)))
            lines += ["", head, "-" * need]
            first = max(args.rows_per_page - len(lines), 8)
            page(pdf, lines + rows[:first], size)
            rest = rows[first:]
            while rest:
                cont = [f"Supplementary Table {num} (continued)", "", head, "-" * need]
                page(pdf, cont + rest[:args.rows_per_page], size)
                rest = rest[args.rows_per_page:]
            print(f"  {num}: {len(d)} rows at {size} pt")

    if "xlsx" in want:
        write_xlsx(outdir, loaded)

    print(f"wrote {pdf_path}  ({pdf_path.stat().st_size / 1024:.0f} KB)")
    if missing:
        print(f"\n{len(missing)} table(s) absent; the PDF was built without them. "
              "Run the missing stage, then rebuild.")
    print("\nThe main text must cite each of these at the point it relies on them; "
          "the journal checks that supplementary data is referenced.")


if __name__ == "__main__":
    main()
