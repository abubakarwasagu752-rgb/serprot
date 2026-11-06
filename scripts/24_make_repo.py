#!/usr/bin/env python
"""Prepare the project as a public repository: licence, README, citation metadata and .gitignore.

Bioinformatics requires the software to be freely available to non-commercial users, with a complete
URL in the Availability section and two years of availability. A GitHub repository archived by Zenodo
satisfies all three and gives a DOI to cite.

This writes the files a public repository needs and stages them, then STOPS. It does not commit
until you pass --commit, and it prints exactly what would be published first -- the point at which
accidentally committing data or credentials becomes irreversible is the point worth pausing at.

  python scripts/24_make_repo.py --config config.yaml \\
      --author "Surname, A.B." --orcid 0000-0000-0000-0000 --github-user yourname
  python scripts/24_make_repo.py --config config.yaml --commit      # once the listing looks right
"""
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from serprot.common import base_parser, load_config  # noqa: E402

TITLE = "Thermal adaptation spares the catalytic neighbourhood in bacterial serine proteases"
BLURB = ("Analysis pipeline for a study of where on a protein fold thermal adaptation acts. "
         "Across 10,674 bacterial and archaeal serine proteases in 1,485 sequence clades, "
         "compositional adaptation to growth temperature is absent within about 10 angstroms of "
         "the catalytic serine and present in a size- and burial-matched control patch on the "
         "same protein.")

GITIGNORE = """\
# Data: gigabytes, and redistributed from their own sources rather than from here.
# DERIVED_INVENTORY.tsv in any freeze snapshot records what was present and when.
data/
freeze/
*.tar
*.tar.gz
*.npy

# Run output other than the analysis tables, which are the paper's numbers and are kept.
runs/*
!runs/analysis/
logs/
nohup.out

# Test harnesses generate whole synthetic project roots; those are output, not source.
tests/synth_*/
tests/*/data/
tests/*/runs/
tests/*/figures/
tests/*/tables/

# Python
__pycache__/
*.pyc
*.pyo
.ipynb_checkpoints/
.venv/
"""

MIT = """\
MIT License

Copyright (c) {year} {holder}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

README = """\
# {title}

{blurb}

The result is bacterial: 628 archaeal proteins in 88 clades cannot resolve the question either way.

## What is here

| Path | Contents |
| --- | --- |
| `scripts/` | The pipeline, one numbered stage per step, each runnable on its own |
| `src/serprot/` | Shared library: structure parsing, SASA, pocket selection, statistics |
| `slurm/` | Batch scripts for each stage, with the resources each one needs |
| `runs/analysis/` | Every analysis table behind the paper, as TSV |
| `figures/` | Publication figures (TIFF at 1200 dpi, plus SVG and PDF) |
| `tables/` | Manuscript tables and the supplementary tables |
| `tests/` | Synthetic positive and negative controls for the main analyses |

Structures, sequence files and the embedding cache are not in this repository: they run to
gigabytes and come from public databases listed below.

## Reproducing the analysis

Stages are numbered in dependency order. Each takes `--config config.yaml` and writes into the
paths that file defines.

```
01_fetch_sequences.py     UniProt sequences for PF00082, PF00326, PF00089
02_build_labels.py        join growth temperatures, locate catalytic serines
03_cluster_split.py       MMseqs2 clustering at 30% and 50% identity
04_embed_baselines.py     ESM-2 650M embeddings and the temperature regressor
09_select_structures.py   choose which proteins need structures
10_fetch_structures.py    AlphaFold DB v4 download
11_pocket_descriptors.py  pocket and matched-control descriptors
12_pocket_analysis.py     within-clade slopes, bootstrap, permutation, FDR
14_residue_attribution.py exact per-residue decomposition of the regressor
15_distance_profile.py    attribution against distance from the catalytic serine
16_burial_distance.py     the same, stratified by solvent exposure
17_esmfold.py             fold the sequences AlphaFold does not cover
18_flexibility.py         elastic network model fluctuations
19_subgroups.py           archaea against bacteria
20_figures.py             publication figures
21_tables.py              manuscript and supplementary tables
22_heldout_eval.py        independent test-set evaluation of the regressor
23_freeze.py              checksummed snapshot of the whole project
```

Sanity checks live in `tests/`. They generate synthetic structures with a known effect planted,
or with none, and confirm the analyses detect the first and return a null on the second.

## Data sources

| Source | Version | Accessed |
| --- | --- | --- |
| UniProt | release 2026_04 | 4 November 2026 |
| AlphaFold Protein Structure Database | v4 | 4-5 November 2026 |
| TEMPURA growth temperatures | current | November 2026 |
| MEROPS peptidase database | 12.1 | November 2026 |
| ESM-2 650M, ESMFold v1 | `facebook/esm2_t33_650M_UR50D`, `facebook/esmfold_v1` | - |

## Requirements

Python 3.11 with the packages in `requirements.txt`, and MMseqs2 on the path. Structure prediction
needs a CUDA GPU; every other stage runs on CPU.

## Citation

If you use this code, please cite the paper and this repository. `CITATION.cff` carries the
metadata, and the Zenodo DOI is minted from the GitHub release.

## Licence

MIT. See `LICENSE`.
"""

CITATION = """\
cff-version: 1.2.0
message: "If you use this software, please cite it using these metadata."
title: "{title}"
abstract: "{blurb}"
authors:
  - family-names: "{family}"
    given-names: "{given}"
{orcid_block}\
license: MIT
repository-code: "https://github.com/{gh_user}/{repo}"
keywords:
  - protein thermostability
  - serine protease
  - protein language model
  - structural bioinformatics
  - extremozymes
date-released: "{today}"
version: "1.0.0"
"""


def write(path: Path, text: str, force: bool, made: list, skipped: list):
    if path.exists() and not force:
        alt = path.with_suffix(path.suffix + ".proposed")
        alt.write_text(text)
        skipped.append((path.name, alt.name))
        return
    path.write_text(text)
    made.append(path.name)


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--author", default="Surname, A.B.", help='"Family, Given" for the citation files')
    ap.add_argument("--orcid", default=None)
    ap.add_argument("--github-user", default="YOUR-GITHUB-USERNAME")
    ap.add_argument("--repo-name", default="serprot")
    ap.add_argument("--holder", default=None, help="copyright holder; defaults to --author")
    ap.add_argument("--commit", action="store_true", help="actually make the first commit")
    ap.add_argument("--force", action="store_true", help="overwrite an existing README.md or LICENSE")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["paths"]["root"]).expanduser().resolve()
    family, _, given = args.author.partition(",")
    orcid_block = f'    orcid: "https://orcid.org/{args.orcid}"\n' if args.orcid else ""
    made, skipped = [], []

    write(root / ".gitignore", GITIGNORE, args.force, made, skipped)
    write(root / "LICENSE", MIT.format(year=date.today().year,
                                       holder=(args.holder or args.author)), args.force, made, skipped)
    write(root / "README.md", README.format(title=TITLE, blurb=BLURB), args.force, made, skipped)
    write(root / "CITATION.cff", CITATION.format(
        title=TITLE, blurb=BLURB, family=family.strip(), given=given.strip(),
        orcid_block=orcid_block, gh_user=args.github_user, repo=args.repo_name,
        today=date.today().isoformat()), args.force, made, skipped)
    write(root / ".zenodo.json", json.dumps({
        "title": TITLE, "description": BLURB, "upload_type": "software",
        "license": "MIT",
        "creators": [{"name": args.author, **({"orcid": args.orcid} if args.orcid else {})}],
        "keywords": ["protein thermostability", "serine protease", "protein language model",
                     "structural bioinformatics", "extremozymes"],
    }, indent=2) + "\n", args.force, made, skipped)

    for n in made:
        print(f"wrote {n}")
    for n, alt in skipped:
        print(f"kept your {n}; proposed version written to {alt}")

    if not (root / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        print("git init")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)

    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=root,
                            capture_output=True, text=True).stdout.split()
    size = sum((root / f).stat().st_size for f in staged if (root / f).exists())
    top = {}
    for f in staged:
        top[f.split("/")[0]] = top.get(f.split("/")[0], 0) + 1
    print(f"\n{len(staged)} files staged, {size / 1024 ** 2:.1f} MB")
    for k in sorted(top, key=lambda k: -top[k]):
        print(f"  {top[k]:>5}  {k}")
    heavy = sorted(((root / f).stat().st_size, f) for f in staged if (root / f).exists())[-5:]
    print("\nlargest staged files:")
    for sz, f in reversed(heavy):
        print(f"  {sz / 1024 ** 2:>8.2f} MB  {f}")

    leaked = [f for f in staged if f.startswith("data/") or f.endswith((".npy", ".tar.gz", ".tar"))]
    if leaked:
        print(f"\nSTOP: {len(leaked)} data files are staged and should not be published:")
        for f in leaked[:10]:
            print(f"  {f}")
        raise SystemExit("fix .gitignore, run `git rm -r --cached <path>`, then try again")

    if not args.commit:
        print("\nNothing committed. Read the listing above, then re-run with --commit.")
        return

    # A public commit carries your name forever; git refuses without one, and a placeholder
    # identity on a repository the paper cites is worse than stopping here.
    who = {k: subprocess.run(["git", "config", f"user.{k}"], cwd=root,
                             capture_output=True, text=True).stdout.strip()
           for k in ("name", "email")}
    if not who["name"] or not who["email"]:
        print("\nSTOP: git has no author identity on this machine. Set it, then re-run:")
        print('  git config --global user.name "Your Name"')
        print('  git config --global user.email "you@institution.edu"')
        raise SystemExit(1)
    print(f"\ncommitting as {who['name']} <{who['email']}>")

    subprocess.run(["git", "commit", "-q", "-m",
                    "Analysis pipeline for thermal adaptation in bacterial serine proteases"],
                   cwd=root, check=True)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                          capture_output=True, text=True).stdout.strip()
    print(f"\ncommitted {head}")
    print("\nNext, with the empty repository already created on GitHub:")
    print(f"  git -C {root} branch -M main")
    print(f"  git -C {root} remote add origin "
          f"https://github.com/{args.github_user}/{args.repo_name}.git")
    print(f"  git -C {root} push -u origin main")


if __name__ == "__main__":
    main()
