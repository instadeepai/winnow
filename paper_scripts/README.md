# Reproducing Winnow paper results and plots

This folder contains the analysis scripts used in the Winnow paper ([arXiv:2509.24952](https://arxiv.org/abs/2509.24952)).
Together with [`Makefile.paper`](../Makefile.paper) in the repository root, they let a reviewer regenerate the reported figures and summary tables from public artefacts.

## 1. Setup

```bash
git clone https://github.com/instadeepai/winnow.git
cd winnow
make -f Makefile.paper paper-sync-group-paper
make -f Makefile.paper help
make -f Makefile.paper paper-setup    # Figshare + HF datasets + models, then artefact check
make -f Makefile.paper paper-plot-light
```

`paper-setup` is the one-shot acquire path.
Be aware that the HF dataset pin is large (Astral and full-search inputs making up the bulk of the size).
The Hugging Face CLI can be used to download specific datasets.
Alternatively, selective downloads are available:

```bash
make -f Makefile.paper download-paper-artefacts
make -f Makefile.paper download-paper-datasets
make -f Makefile.paper download-paper-models
make -f Makefile.paper paper-check-artefacts
```

Recompute umbrellas (after setup):

```bash
make -f Makefile.paper paper-recompute-laptop   # no large datasets / no GPU
make -f Makefile.paper paper-recompute-heavy    # large datasets, GPU
```

Or run one analysis at a time (`paper-plot-helaqc-analysis`,
`paper-recompute-external-peptide-holdout`, …).
See `make -f Makefile.paper help`.

`winnow predict` for general-model and HeLa recomputes calls Koina (public server via `koinapy`) for fragment-intensity and iRT features.
Outbound network access is required and runtime is likely to be dominated by these calls.

### Command naming: `plot` vs `recompute`

| Prefix | Means | May call |
| --- | --- | --- |
| `paper-plot-*` | Build figures / summary tables from Figshare deposits (or a deterministic local transform of them). | CPU only; never Koina-dependent, never uses calibrator training or inference |
| `paper-recompute-*` | Rebuild intermediates that the deposit does not fully pin: inference, Koina feature computation, calibrator training, or timed benchmarks. Scripts often write plots at the end of the same target. | Koina and/or GPU as noted |

If a paper figure has no deposited artefact that can be plotted without those steps, there is no `paper-plot-*` target (table cell says “n/a”). The `paper-recompute-*` target still produces the plots when it finishes.

### Umbrella membership

Laptop vs heavy is about workload size and hardware requirements.

| Umbrella | Rule | Contents |
| --- | --- | --- |
| `paper-setup` | acquire | `download-paper-artefacts`, `download-paper-datasets`, `download-paper-models`, `paper-check-artefacts` |
| `paper-plot-light` | deposit replot without the largest datasets | HeLa DNS, general labelled, FDR method / holdout, generalisation heatmap, feature importance, novelty, upscored FPs, FDR overlap (**immuno2** only) |
| `paper-recompute-laptop` | no GPU; no unlabelled / large cohorts | HeLa DNS predict, general **labelled** (default stems), feature investigation, FDR method / holdout for **helaqc** |
| `paper-recompute-heavy` | large cohorts, GPU | `general-full-*`, feature-importance, generalisation, ablations, runtime, scaling, FDR tools including **celegans**, `paper-plot-general-full`, FDR overlap (**all** projects) |

## 2. Public pins

| Resource | Pin |
| --- | --- |
| Figshare [Analysis outputs](https://figshare.com/articles/dataset/Analysis_outputs/30147601) | article `30147601` **v7** (`10.6084/m9.figshare.30147601.v7`) |
| Hugging Face [winnow-ms-datasets](https://huggingface.co/datasets/InstaDeepAI/winnow-ms-datasets) | `659802319d618a359de5ab90ec6b0195681e94a6` |
| Hugging Face [winnow-general-model](https://huggingface.co/InstaDeepAI/winnow-general-model) | `e2089330dd59adb9685e5b3d7d61f0cd69a3bbb0` |
| Hugging Face [winnow-helaqc-model](https://huggingface.co/InstaDeepAI/winnow-helaqc-model) | `d56542b961eac7d896e51bf0716a242fc394ab1f` |
| Figshare [Additional HeLa Single Shot models](https://doi.org/10.6084/m9.figshare.32744946.v2) | `32744946` v2 (Casanovo / π-PrimeNovo calibrators) |
| Glissade ([JemmaLDaniel/glissade](https://github.com/JemmaLDaniel/glissade), branch `winnow-benchmark`) | `7c723a2af4a88fda84a6bd4f223b351179bd36da` via `uv sync --group paper` |
| NovoBoard (only if regenerating decoy CSVs from MGFs) | `a9faab3ef1af06987599c2f01e6ba96072c80172` |

## 3. What each analysis does

| Paper analysis | Plot (from Figshare) | Recompute | Needs | Script |
| --- | --- | --- | --- | --- |
| HeLa Single Shot DNS (InstaNovo, Casanovo, π-PrimeNovo × test / raw_less_train) | `paper-plot-helaqc-analysis` | `paper-recompute-helaqc` | Koina | `plot_analysis.py` |
| General-model evaluation (nine projects, labelled and full-search) | `paper-plot-general-labelled`, `paper-plot-general-full` | tiered / per-stem `paper-recompute-general-*` | Koina | `plot_eval_results.py` |
| Feature investigation (InstaNovo HeLa) | n/a (matrices not deposited) | `paper-recompute-feature-investigation` (compute + plot) | Koina | `plot_feature_investigation.py` |
| General-model feature importance (*C. elegans*) | `paper-plot-feature-importance` (no SHAP bar / correlations) | `paper-recompute-feature-importance` | Koina | `analyze_features.py` |
| Pipeline scaling excluding Koina features | n/a (JSON not deposited; script writes plots) | `paper-recompute-scaling` | GPU + Koina | `benchmark_scaling.py` |
| Pipeline runtime table (full + no-Prosit) | n/a (JSON not deposited) | `paper-recompute-runtime` | GPU + Koina | `benchmark_runtime.py` |
| PSM-level FDR vs NovoBoard | `paper-plot-fdr-method-comparison` (`--summarise-only`) | `paper-recompute-fdr-method-comparison` | CPU | `plot_fdr_method_comparison.py` |
| External peptide score-mixture (Winnow / NovoBoard / Glissade) | `paper-plot-external-peptide-holdout` | `paper-recompute-external-peptide-holdout` | CPU | `run_external_peptide_holdout_benchmark.py` |
| Feature ablations | n/a (deposit lacks tail ECE for top 10% PSMs; script writes plots) | `paper-recompute-ablations` | GPU + Koina | `run_feature_ablations.py` |
| Calibrator generalisation heatmap | `paper-plot-generalisation` | `paper-recompute-generalisation` | GPU + Koina | `evaluate_calibrator_generalisation.py`, `plot_calibrator_generalisation_heatmap.py` |
| Upscored FPs | `paper-plot-upscored-fps` | n/a (uses deposited labelled `general_results/`) | CPU | `analyze_upscored_fps.py` |
| FDR overlap | `paper-plot-fdr-overlap` | n/a (uses deposited labelled + full trees) | CPU | `analyze_fdr_overlap.py` |
| Novelty | `paper-plot-novelty` | n/a (uses deposited chymotrypsin / ProteomeTools trees) | CPU | `analyze_novelty.py` |

### General-model projects (`plot_eval_results`)

Make / CLI `--projects` use leaf folder names.
Nested Figshare trees are resolved automatically (`PXD004452/<run>/` or flat `PXD004732/`):

- `20150708_QE3_UPLC8_DBJ_QC_HELA_39frac_Chymotrypsin`
- `20151020_QE3_UPLC8_DBJ_SA_A549_Rep2_46`
- `PXD004732`
- `20170303_QEh1_LC2_FaMa_ChCh_SA_HLApI_JY_R1_exp2`
- `20170609_QEh1_LC1_ChCh_FAMA_SA_HLAIIp_JY_all_R1`
- `01747_C01_P018218_S00_I00_N03_R1`
- `PXD014877`
- `PXD023064`
- `astral`

### General-model recompute (HF map and Make targets)

Inputs live under `paper_data/winnow-ms-datasets/general_model_evaluation/<stem>/{labelled,full}/`.
Outputs mirror Figshare under `paper_results/general_results/{labelled,full}/<figshare_key>/`.

| HF stem (Make suffix) | Figshare project key | FASTA |
| --- | --- | --- |
| `hela_chymotrypsin` | `PXD004452/20150708_…_Chymotrypsin` | `fasta/human.fasta` |
| `human_lung` | `PXD004452/20151020_…_A549_Rep2_46` | `fasta/human.fasta` |
| `proteometools1` | `PXD004732` | `fasta/human.fasta` |
| `HLA_I` | `PXD006939/…_HLApI_…` | `fasta/human.fasta` |
| `HLA_II` | `PXD006939/…_HLAIIp_…` | `fasta/human.fasta` |
| `athaliana` | `PXD013868/01747_…` | `fasta/athaliana.fasta` |
| `celegans` | `PXD014877` | `fasta/celegans.fasta` |
| `immuno2` | `PXD023064` | `fasta/human.fasta` |
| `astral` | `astral` | `fasta/ecoli_zorya.fasta` |

**Labelled:** `winnow predict` only.

**Full-search:** `winnow predict`, then `annotate_preds_proteome_hits.py` (post-predict; adds `proteome_hit`, drops peptides shorter than 7 residues).

Note that **immuno2** always subsets to the Figshare cohort (`PXD023064_FILES`) before predict.

All paper `winnow predict` calls pass `fdr_control.fdr_threshold=1.0` (`PREDICT_FDR_THRESHOLD`) so outputs keep every PSM with FDR columns.

Koina collision energy / fragmentation:

- **HeLa Single Shot** (`paper-recompute-helaqc`, feature investigation): tiled constants `collision_energies=27`, `fragmentation_types=HCD` (`KOINA_FRAGMENT_MATCH_CONSTANTS`).
- **General-model evaluation** (most HF stems): null those constants (`KOINA_FRAGMENT_MATCH_COLUMNS`) so runtime resolution uses the default metadata columns `collision_energy` / `frag_type` (see `docs/configuration.md` and `resolve_feature_model_inputs`).
- **immuno2 / PXD023064**: tiled `CE=27` / `HCD`.

Batch / print:

```bash
make -f Makefile.paper paper-recompute-general-labelled          # default stems: immuno2 hela_chymotrypsin human_lung
make -f Makefile.paper paper-recompute-general-full-small        # in paper-recompute-heavy
make -f Makefile.paper paper-recompute-general-full-large        # in paper-recompute-heavy
make -f Makefile.paper paper-recompute-general-print             # echo all nine × labelled/full commands
```

Per dataset (HF stem):

```bash
make -f Makefile.paper paper-recompute-general-labelled-immuno2
make -f Makefile.paper paper-recompute-general-full-immuno2
make -f Makefile.paper paper-recompute-general-immuno2           # labelled + full
```

Override labelled batch: `GENERAL_RECOMPUTE_STEMS='immuno2 athaliana' make -f Makefile.paper paper-recompute-general-labelled`.

HeLa DNS tools default to all three (`HELAQC_DNS_TOOLS=instanovo casanovo primenovo`). Subset with e.g. `HELAQC_DNS_TOOLS=instanovo`.

Selective HF download (skip Astral when you only need a small project):

```bash
uv run hf download InstaDeepAI/winnow-ms-datasets \
  --repo-type dataset \
  --revision 659802319d618a359de5ab90ec6b0195681e94a6 \
  --include 'general_model_evaluation/immuno2/**' \
  --include 'fasta/**' \
  --local-dir paper_data/winnow-ms-datasets
```

### How full-search proteome annotation differs from the package CLI

Paper full recompute intentionally keeps the old post-predict path so outputs stay aligned with deposited `general_results/full/` artefacts.
The package CLI `winnow annotate-proteome-hits` is the supported product path going forward (annotate-then-predict / diagnose).
Reviewers who follow only `docs/cli.md` will get a different order and can differ slightly from deposits.

| | Paper helper (this suite) | Package CLI |
| --- | --- | --- |
| Command | `paper_scripts/annotate_preds_proteome_hits.py` | `winnow annotate-proteome-hits` |
| When | **After** `winnow predict` | **Before** predict (or for diagnose) |
| Input | Predict output folder | Spectra + de novo preds via `DatasetLoader` |
| Output | In-place CSVs with `proteome_hit`; short peptides removed | New Winnow dataset |
| FDR | Estimated on all PSMs, then short peptides dropped | Short peptides dropped before feature/FDR if you annotate then predict |
| Used for | Reproducing Figshare `general_results/full/` | Diagnose / annotate-then-predict workflows in main docs |

## 4. Approximate deposit sizes

Orders of magnitude for Figshare v7 outputs (metadata + prediction CSVs unless noted):

| Tree | Size |
| --- | --- |
| `general_results/labelled/` (nine projects) | ~2 GB |
| `general_results/full/` | ~5 GB (astral ~2 GB; *C. elegans* ~1.8 GB) |
| `fdr_benchmark_inputs/` | ~1 GB+ (includes HeLa MGFs for twin pairing) |
| Generalisation results CSV | ~6.8 GB |
| Remaining analysis CSVs / HeLa result trees | much smaller |

Approximate HF input sizes (parquet + InstaNovo preds) for recompute:

| Tier | Stems | Notes |
| --- | --- | --- |
| Labelled defaults | `immuno2`, `hela_chymotrypsin`, `human_lung` | Tens–hundreds of MB each |
| Full small | `hela_chymotrypsin`, `human_lung`, `HLA_I/II`, `athaliana`, `immuno2` | immuno2 full ~1.4 GB inputs |
| Cluster | `astral` full and labelled, `celegans` full, `proteometools1` full | Astral labelled ~2.7 GB; Astral full ~6.7 GB inputs |

Replotting labelled deposits and `paper-plot-light` is the most accessible review path.
`paper-plot-general-full` (and full FDR overlap for large datasets) are slow, so they are listed under `paper-recompute-heavy` even though they are still `paper-plot-*` targets.
Use `paper-recompute-general-print` to obtain the exact cluster predict commands without running them.

## 5. FDR tool benchmarking

For both PSM-level comparison and the peptide holdout:

1. Download `fdr_benchmark_inputs/` with the Figshare article (or `paper-setup` / `download-paper-artefacts`).
2. Replot with `--summarise-only` on the deposited curves / results CSVs, or recompute with `--novoboard-root paper_data/fdr_benchmark_inputs/novoboard` and the matching Winnow prediction folders.

NovoBoard’s twin-decoy competition is reimplemented here against those CSVs. Glissade’s bootstrap FDR is imported from the package installed by `uv sync --group paper`.
We do not cover reproducing decoy spectra here, but this can be done using standard NovoBoard decoy generation commands and the per-dataset decoy generation strategies described in the paper.

## 6. Entrypoints

| Script | Role |
| --- | --- |
| `plot_analysis.py` | HeLa DNS evaluation plots for InstaNovo, Casanovo and π-PrimeNovo |
| `plot_eval_results.py` | General model evaluation figures |
| `plot_feature_investigation.py` | Feature distributions / investigation (after recompute matrices) |
| `benchmark_scaling.py` | Runtime vs dataset size (trains/reuses a no-Prosit dummy, then times the pipeline) **[GPU]** |
| `benchmark_runtime.py` | Stage-wise wall-time / memory table (full Prosit + no-Prosit; same dummy as scaling) **[GPU]** |
| `no_prosit_dummy.py` | Shared train-or-reuse helper for the no-Prosit dummy calibrator |
| `plot_fdr_method_comparison.py` | Winnow vs NovoBoard PSM FDR |
| `run_external_peptide_holdout_benchmark.py` | Controlled-π₀ peptide mixture benchmark |
| `run_feature_ablations.py` | Feature-subset calibrator training + eval **[GPU]** |
| `plot_ablation_summary.py` | Ablation bar summaries (from recompute outputs, not Figshare alone) |
| `plot_calibrator_generalisation_heatmap.py` | Hold-one-out generalisation heatmap |
| `evaluate_calibrator_generalisation.py` | Full retrain leave-one-source-out results CSV **[GPU]** |
| `analyze_features.py` | Feature importance / SHAP (supports `--replot-dir` from Figshare pickles) |
| `analyze_upscored_fps.py`, `analyze_fdr_overlap.py`, `analyze_novelty.py` | Downstream analyses of deposited `general_results/` |
| `annotate_preds_proteome_hits.py` | Post-predict proteome hits for full-search recompute |
| `download_figshare_article.py` | Figshare download with folder layout |
| `subset_eval_by_experiment.py` | Filter immuno2 (and similar) to the deposited cohort |

Every entrypoint supports `--help`.
