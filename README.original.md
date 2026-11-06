# serprot

**Separating the Thermal and Specificity Signals in Serine Protease Sequences: A Cross-Family Test of the Stability-Specificity Trade-Off**

A pipeline that (1) builds a lineage-controlled dataset of prokaryotic S8 / S9 / S1 serine proteases, (2) trains a
temperature-aware ESM-2 model with separate thermal and specificity latents, and (3) tests whether
thermal adaptation narrows substrate specificity, within clades of relatives and at the residue level.

## Stages

| # | script | what it does | needs |
|---|---|---|---|
| 0 | `00_check_access.py` | pre-flight: UniProt / AlphaFold / Hugging Face reachable, files present, GPU visible | internet |
| 1 | `01_fetch_sequences.py` | prokaryotic sequences for Pfam PF00082 (S8), PF00326 (S9), PF00089 (S1); catalytic Ser from UniProt `ACT_SITE`, motif fallback | internet |
| 2 | `02_build_labels.py` | joins OGT (and optional Tm), turns cleavage windows into per-protease specificity profiles + breadth | `ogt_table.csv`, `cleavages.tsv` |
| 3 | `03_cluster_split.py` | MMseqs2 clusters at 30% (held-out groups) and 50% (clades); stratified group CV + leave-one-family-out | mmseqs2 |
| 4 | `04_embed_baselines.py` | physicochemical and frozen-ESM ridge baselines on the same folds | GPU (ESM part) |
| 5 | `05_train_disentangled.py` | dual-subspace ESM-2 (variants: joint, plain, disent, noadv, nodcor), one fold per call | GPU |
| 6 | `06_analyze.py` | accuracy, latent leakage, within-clade trade-off slopes (observed and predicted), figures | - |
| 7 | `07_attribution_pockets.py` | residue attribution for both heads, pocket enrichment against size-matched controls, overlap of the two heads | GPU, AlphaFold DB or `--pocket-file` |
| 8 | `08_candidates.py` | 2x2 (hot/cool x narrow/broad) panels of natural homologs for wet-lab validation | - |

## Run it

```bash
# one-time
python -m pip install -r requirements.txt          # plus mmseqs2 (conda/module)
# edit config.yaml: paths.root -> your big shared folder; esm_model -> esm2_t33_650M_UR50D once the 35M run works
# edit slurm/env.sh: module loads / environment activation

# put your two label tables in <root>/data/raw/  (formats below), then:
PARTITION=<gpu partition> ACCOUNT=<account> bash slurm/run_all.sh
```

Smoke test first (small, ~minutes; uses a tiny random ESM and synthetic data with a planted trade-off):
`bash tests/smoke_test.sh`


## CPE-KMUTT cluster quick start (L40 / 4090 GPUs, 12 h limit, 100 GB /home quota)

```bash
bash slurm/preflight.sh 2>&1 | tee preflight.txt     # login node, read-only
bash slurm/setup_env.sh                              # login node, one-time conda env + mmseqs2
sbatch slurm/smoke_gpu.sbatch                        # 20 min: GPU + 650M memory/speed check
# then follow the stage-by-stage order in the project chat (data -> labels -> baselines -> train -> analyse)
```

Jobs must go through sbatch/srun (no computation on the login node). Keep an eye on `/home` quota (100 GB): the 650M model
cache is ~2.6 GB and each trained fold checkpoint is a few GB, so delete `runs/*/fold*/model.pt` for runs you do not need.

## Input files you must supply

1. **`data/raw/ogt_table.csv`**: optimal growth temperatures per organism. Columns: `taxid` (optional) and/or `organism`, plus `ogt`.
   TEMPURA (Sato et al. 2020, *Microbes Environ.* 35:ME20074) is the natural source; I could not verify its download URL from my
   workspace, so export it from the database yourself. Matching is by NCBI taxid first, then by binomial name; species whose
   entries disagree by more than 10 C are dropped.
2. **`data/raw/cleavages.tsv`**: tab-separated `protease_acc` (UniProt accession) and `window` (8 residues P4 P3 P2 P1 P1' P2' P3' P4'; use `-` for gaps).
   MEROPS holds this information but I could **not** confirm a bulk download for substrate cleavage data (the documentation only
   mentions web views per peptidase). Options: export from MEROPS pages, ask the MEROPS maintainers for the substrate table, or use
   peptide-library / proteomic profiling data (e.g. PICS, TAILS) from papers. Anything that produces P4..P4' windows works.
3. *(optional)* **`data/raw/tm_table.csv`**: `accession`, `tm` (e.g. from the Meltome Atlas / ProThermDB / the TmProt Zenodo sets).
   Adds a second thermal label and a real stability signal beyond host growth temperature.

## Design choices that matter scientifically

- **Lineage is the main confounder.** Held-out groups are 30%-identity clusters, so test proteins have no close relatives in
  training. The trade-off tests use *within-clade* contrasts (cluster fixed effects at 50% identity), so they compare relatives
  that differ in growth temperature rather than distant lineages.
- **OGT is a proxy for stability.** It labels the *host*, not the enzyme. The optional Tm table and the planned wet-lab panel
  (stage 8) are the only direct checks. Say this explicitly in any write-up.
- **Specificity = P4..P2' profile + breadth.** Breadth is mean normalised Shannon entropy across positions (0 strict, 1 flat),
  computed from smoothed counts; proteases with fewer than `min_cleavages` observed cleavages get no label and sample weights
  scale with the count. Breadth from small samples is noisy, so keep `min_cleavages` honest.
- **Disentanglement is a measurement, not a guarantee.** The adversaries and the distance-correlation penalty push the two
  latents apart; if the trade-off is real, forcing independence should *cost* specificity accuracy, and the `joint` vs `disent`
  comparison (stage 6) quantifies that cost. Do not read lower leakage alone as "the signals are independent".
- **Predicted slopes are attenuated.** Model-predicted breadth regresses toward the mean, so the predicted within-clade slope is
  usually smaller in magnitude than the observed one (visible in the synthetic test). Use the observed-label slope as the primary result and the predicted slope to extend
  coverage to unlabeled proteins.
- **Pockets are coarse.** The default pocket is every residue within 8 A of the catalytic Ser OG in the AlphaFold model, compared against
  size-matched control regions. For S1-S4 resolution supply substrate-complex pockets via `--pocket-file`.

## Data caveats to expect

- The number of proteases with enough cleavage data is likely in the low hundreds, and thin for non-S8 families. The model is
  multi-task with masked labels (temperature from tens of thousands of sequences, specificity from the small labeled subset), but
  family-level specificity results may be underpowered. `02_build_labels.py` prints the counts per family.
- If `ogt_match` is mostly `species`, strain-level temperature differences are lost.
- S1 (trypsin-like) is rare in prokaryotes; `max_per_family` and the Pfam queries are in `config.yaml`.
- The Pfam accessions (PF00082, PF00326, PF00089) and the UniProt query syntax are from my notes; run `00_check_access.py` and check the
  per-family totals look sensible before launching anything expensive.

## What was tested here

Everything except the live downloads (UniProt, AlphaFold DB, Hugging Face, MMseqs2) was exercised on synthetic data. The synthetic
test plants a known trade-off and checks the observed within-clade slope recovers it. Real-data behaviour (accuracy, runtimes, memory
for the 650M model) has not been observed.

## Outputs

`data/processed/dataset.tsv` (master table), `runs/<variant>_<split>/fold*/` (models, predictions, latents),
`runs/analysis/` (performance, leakage, trade-off slopes, attribution stats, candidates), `figures/*.png`.
