# Thermal adaptation spares the catalytic neighbourhood in bacterial serine proteases

Analysis pipeline for a study of where on a protein fold thermal adaptation acts. Across 10,674 bacterial and archaeal serine proteases in 1,485 sequence clades, compositional adaptation to growth temperature is absent within about 10 angstroms of the catalytic serine and present in a size- and burial-matched control patch on the same protein.

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
