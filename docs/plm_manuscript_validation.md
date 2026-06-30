# PLM Manuscript-Style Validation Report

Date: 2026-06-28

Branch: `feat/plm-manuscript-validation`

## Objective

Prototype a manuscript-style validation run for `PeptideLanguageModelFeature` using
PepBERT as the first backend, comparing the existing confidence-only calibrator
against confidence plus PLM embedding-summary features.

## Aichor Runs

| Experiment ID | Status | Notes |
| --- | --- | --- |
| `be117e33-b468-4c92-941a-6a7675e551de` | Failed | Initial run used flattened local dataset names that did not match the Hugging Face dataset layout. |
| `b47b5f40-d135-4718-8b32-467cbff0fb07` | Cancelled | Corrected dataset paths, but PepBERT was scoring sequences one at a time on CPU and was too slow. |
| `56e7e93d-a215-4e9f-8f34-058e45fdf129` | Succeeded | Batched PepBERT scoring and progress logging. |
| `398417c5-73a0-46da-9475-fcc1afc4708f` | Succeeded | Full uncapped manuscript-style validation run. |

## Successful Run Settings

The successful run used the manuscript dataset layout from
`InstaDeepAI/winnow-ms-datasets`.

| Setting | Value |
| --- | --- |
| Training set | HeLa QC train |
| Evaluation sets | HeLa QC test, celegans labelled, immuno2 labelled |
| Train cap | 3000 PSMs |
| Eval cap | 3000 PSMs, except HeLa QC test used all 1768 PSMs |
| PLM backend | PepBERT |
| PLM feature mode | `embedding_summary` |
| FDR thresholds | 0.05, 0.10 |

## Dataset Sizes

| Dataset | Loaded PSMs | Used PSMs |
| --- | ---: | ---: |
| HeLa QC train | 14147 | 3000 |
| HeLa QC test | 1768 | 1768 |
| celegans labelled | 232156 | 3000 |
| immuno2 labelled | 20002 | 3000 |

## Calibration Metrics

| Dataset | Variant | PR-AUC | Brier | ECE |
| --- | --- | ---: | ---: | ---: |
| HeLa QC test | confidence only | 0.9614 | 0.0925 | 0.0786 |
| HeLa QC test | confidence + PepBERT | 0.9637 | 0.0875 | 0.0198 |
| celegans | confidence only | 0.8738 | 0.1456 | 0.0666 |
| celegans | confidence + PepBERT | 0.8699 | 0.1446 | 0.0628 |
| immuno2 | confidence only | 0.8548 | 0.1997 | 0.2200 |
| immuno2 | confidence + PepBERT | 0.7604 | 0.1949 | 0.1167 |

## FDR-Control Results

The confidence-only variant accepted 0 PSMs at both 5% and 10% FDR in this capped
run because the fitted FDR range did not reach those thresholds.

| Dataset | Variant | Threshold | Accepted PSMs | Realized FDR | Confidence Cutoff | TECE | sTECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HeLa QC test | confidence + PepBERT | 0.05 | 1060 | 0.0509 | 0.7850 | 0.0185 | -0.0010 |
| HeLa QC test | confidence + PepBERT | 0.10 | 1261 | 0.0991 | 0.3801 | 0.0208 | 0.0006 |
| celegans | confidence + PepBERT | 0.05 | 1141 | 0.1174 | 0.8748 | 0.0675 | -0.0675 |
| celegans | confidence + PepBERT | 0.10 | 1706 | 0.1676 | 0.7141 | 0.0677 | -0.0677 |
| immuno2 | confidence + PepBERT | 0.05 | 29 | 0.0345 | 0.9364 | 0.0164 | 0.0154 |
| immuno2 | confidence + PepBERT | 0.10 | 125 | 0.0320 | 0.8315 | 0.0706 | 0.0680 |

## Interim Interpretation

The capped run shows that the PLM feature path is technically functional in the
container and can improve calibration metrics on some datasets. The strongest
calibration improvement is on HeLa QC test and immuno2 ECE. However, the result is
not uniformly positive: immuno2 PR-AUC drops with PepBERT, and celegans realized
FDR exceeds the target thresholds. These interim results should be treated as a
prototype sanity check, not claim-ready evidence.

## Follow-Up

The next run should remove the row caps:

```bash
uv run python scripts/plm_manuscript_validation.py \
  --datasets helaqc,celegans,immuno2 \
  --max-train-rows 0 \
  --max-eval-rows 0 \
  --plm-batch-size 128 \
  --output-dir /runs/plm-validation-full
```

## Full Uncapped Re-Run

The full uncapped run was submitted to Aichor after adding a robustness patch for
backend-missing PLM features, so PepBERT sequences that exceed the model sequence
limit are marked as missing and imputed when `learn_from_missing=True` instead of
terminating the job.

| Setting | Value |
| --- | --- |
| Experiment ID | `398417c5-73a0-46da-9475-fcc1afc4708f` |
| Submission message | `PLM manuscript-style validation full uncapped PepBERT` |
| Output directory | `/runs/plm-validation-full` |
| Train cap | uncapped (`--max-train-rows 0`) |
| Eval cap | uncapped (`--max-eval-rows 0`) |
| PLM batch size | 128 |
| Aichor resources | 8 CPU, 32 GB RAM |
| Timeout | 1 day |
| Status | Succeeded |
| Runtime | 20m57s |

Pre-submit checks passed:

```bash
uv run pytest tests/calibration/features/test_peptide_language_model.py
uv run --extra plm python scripts/plm_manuscript_validation.py \
  --data-dir /tmp/winnow-plm-validation-remote-smoke/data \
  --output-dir /tmp/winnow-plm-validation-remote-smoke/out-full-preflight \
  --datasets helaqc,immuno2 \
  --max-train-rows 6 \
  --max-eval-rows 6 \
  --plm-batch-size 3
```

The run completed successfully and wrote outputs to `/runs/plm-validation-full`.

| Dataset | Loaded PSMs |
| --- | ---: |
| HeLa QC train | 14147 |
| HeLa QC test | 1768 |
| celegans labelled | 232156 |
| immuno2 labelled | 20002 |

The HeLa QC train set contained 9903 correct PSMs out of 14147.

### Full Calibration Metrics

| Dataset | Variant | PR-AUC | Brier | ECE |
| --- | --- | ---: | ---: | ---: |
| HeLa QC test | confidence only | 0.9636 | 0.0872 | 0.0368 |
| HeLa QC test | confidence + PepBERT | 0.9632 | 0.0856 | 0.0174 |
| celegans | confidence only | 0.8856 | 0.1470 | 0.0642 |
| celegans | confidence + PepBERT | 0.8705 | 0.1480 | 0.0660 |
| immuno2 | confidence only | 0.8660 | 0.1942 | 0.2203 |
| immuno2 | confidence + PepBERT | 0.8014 | 0.1817 | 0.1308 |

### Full FDR-Control Results

| Dataset | Variant | Threshold | Accepted PSMs | Realized FDR | Confidence Cutoff | TECE | sTECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HeLa QC test | confidence only | 0.05 | 994 | 0.0352 | 0.8530 | 0.0216 | 0.0147 |
| HeLa QC test | confidence only | 0.10 | 1222 | 0.0876 | 0.5349 | 0.0233 | 0.0124 |
| HeLa QC test | confidence + PepBERT | 0.05 | 1031 | 0.0485 | 0.8036 | 0.0146 | 0.0015 |
| HeLa QC test | confidence + PepBERT | 0.10 | 1253 | 0.0958 | 0.5063 | 0.0158 | 0.0042 |
| celegans | confidence only | 0.05 | 75356 | 0.1015 | 0.9177 | 0.0535 | -0.0515 |
| celegans | confidence only | 0.10 | 125930 | 0.1574 | 0.6935 | 0.0587 | -0.0574 |
| celegans | confidence + PepBERT | 0.05 | 80357 | 0.1113 | 0.8811 | 0.0613 | -0.0613 |
| celegans | confidence + PepBERT | 0.10 | 125322 | 0.1630 | 0.7324 | 0.0630 | -0.0630 |
| immuno2 | confidence only | 0.05 | 167 | 0.0000 | 0.9408 | 0.0500 | 0.0500 |
| immuno2 | confidence only | 0.10 | 885 | 0.0316 | 0.8175 | 0.0683 | 0.0683 |
| immuno2 | confidence + PepBERT | 0.05 | 172 | 0.0174 | 0.9245 | 0.0324 | 0.0324 |
| immuno2 | confidence + PepBERT | 0.10 | 651 | 0.0261 | 0.8480 | 0.0738 | 0.0738 |

### Full-Run Interpretation

The full uncapped run does not support a simple claim that PepBERT improves
Winnow across datasets. The strongest positive signal is on HeLa QC test, where
PepBERT reduces Brier score, roughly halves ECE, improves TECE/sTECE, and accepts
slightly more PSMs while staying near the target FDR thresholds. On immuno2,
PepBERT improves Brier score and ECE but substantially reduces PR-AUC and accepts
fewer PSMs at 10% FDR. On celegans, PepBERT worsens PR-AUC, Brier score, ECE,
TECE/sTECE, and realized FDR relative to confidence-only.

The prototype is technically viable, but these results argue for treating PLM
features as experimental until feature construction, model choice, and
domain-transfer behavior are better understood.

## In-Domain Split Follow-Up

The next diagnostic is to test whether the poor celegans and immuno2 transfer
results are caused by the PLM feature itself or by training the calibrator only
on HeLa QC. The validation script now supports:

```bash
uv run python scripts/plm_manuscript_validation.py \
  --validation-mode in_domain_split \
  --datasets celegans,immuno2 \
  --split-train-fraction 0.5 \
  --max-eval-rows 0 \
  --plm-batch-size 128 \
  --output-dir /runs/plm-validation-indomain
```

This mode loads each selected labelled dataset, optionally caps it with
`--max-eval-rows`, randomly splits the selected rows into train and held-out
evaluation partitions, then runs the same confidence-only versus confidence plus
PepBERT comparison. It writes the same `metrics.json` and `summary.csv` outputs,
with explicit `validation_mode`, `train_dataset`, and `eval_dataset` columns in
the summary.

Expected interpretation:

| Outcome | Interpretation |
| --- | --- |
| PepBERT improves in-domain celegans/immuno2 | The HeLa-trained model likely failed because of domain transfer. |
| PepBERT still hurts in-domain celegans/immuno2 | The current PepBERT feature construction is probably not adding useful sequencing-error signal. |
| PepBERT helps one dataset but not the other | Backend/model choice or peptide distribution differences need dataset-specific ablation. |

### In-Domain Smoke Test

A local smoke test passed with 12 sampled rows per dataset and a 50/50 split:

```bash
uv run --extra plm python scripts/plm_manuscript_validation.py \
  --validation-mode in_domain_split \
  --data-dir /tmp/winnow-plm-validation-remote-smoke/data \
  --output-dir /tmp/winnow-plm-validation-remote-smoke/out-indomain-smoke \
  --datasets celegans,immuno2 \
  --max-eval-rows 12 \
  --split-train-fraction 0.5 \
  --plm-batch-size 3
```

The smoke confirms that the new mode downloads the selected external datasets,
builds held-out splits, fits both calibrator variants per dataset, and writes
`metrics.json` plus `summary.csv`. The row count is too small for scientific
interpretation; its only purpose is checking the workflow.

### Full In-Domain Run

The full in-domain celegans/immuno2 run was submitted to Aichor:

| Setting | Value |
| --- | --- |
| Experiment ID | `bf25368d-716a-4737-aff2-cffad08e0ee7` |
| Submission message | `PLM in-domain split validation PepBERT celegans immuno2` |
| Validation mode | `in_domain_split` |
| Datasets | celegans, immuno2 |
| Split | 50% train, 50% held-out evaluation |
| Row caps | none |
| Output directory | `/runs/plm-validation-indomain` |
| Status | Succeeded |
| Runtime | 17m58s |

The run completed successfully and wrote outputs to
`/runs/plm-validation-indomain`.

| Dataset | Split | PSMs | Correct PSMs |
| --- | --- | ---: | ---: |
| celegans | train | 116078 | 68425 |
| celegans | held-out eval | 116078 | 68265 |
| immuno2 | train | 10001 | 4012 |
| immuno2 | held-out eval | 10001 | 4152 |

#### In-Domain Calibration Metrics

| Dataset | Variant | PR-AUC | Brier | ECE |
| --- | --- | ---: | ---: | ---: |
| celegans | confidence only | 0.8864 | 0.1409 | 0.0105 |
| celegans | confidence + PepBERT | 0.8881 | 0.1383 | 0.0048 |
| immuno2 | confidence only | 0.8675 | 0.1308 | 0.0872 |
| immuno2 | confidence + PepBERT | 0.8517 | 0.1302 | 0.0315 |

#### In-Domain FDR-Control Results

| Dataset | Variant | Threshold | Accepted PSMs | Realized FDR | Confidence Cutoff | TECE | sTECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| celegans | confidence only | 0.05 | 0 | n/a | n/a | n/a | n/a |
| celegans | confidence only | 0.10 | 39085 | 0.1038 | 0.8218 | 0.0099 | -0.0038 |
| celegans | confidence + PepBERT | 0.05 | 4289 | 0.0368 | 0.9448 | 0.0134 | 0.0132 |
| celegans | confidence + PepBERT | 0.10 | 36759 | 0.0962 | 0.8204 | 0.0063 | 0.0038 |
| immuno2 | confidence only | 0.05 | 1145 | 0.0638 | 0.8927 | 0.0168 | -0.0138 |
| immuno2 | confidence only | 0.10 | 2099 | 0.1034 | 0.8064 | 0.0184 | -0.0034 |
| immuno2 | confidence + PepBERT | 0.05 | 1229 | 0.0822 | 0.8902 | 0.0337 | -0.0322 |
| immuno2 | confidence + PepBERT | 0.10 | 2328 | 0.1130 | 0.7993 | 0.0176 | -0.0130 |

#### In-Domain Interpretation

The in-domain split changes the celegans conclusion. Under HeLa transfer,
PepBERT worsened celegans. When trained and evaluated on held-out celegans
splits, PepBERT modestly improves PR-AUC, Brier score, ECE, and 10% FDR tail
calibration, and it enables a finite 5% FDR cutoff where confidence-only could
not estimate one. This suggests that celegans failure in the manuscript-style
transfer run was largely a domain-transfer problem, not a complete lack of PLM
signal.

The immuno2 result remains mixed. PepBERT improves Brier score slightly and ECE
substantially, but PR-AUC drops and realized FDR becomes worse at both 5% and
10%. That points to a ranking/tail-calibration issue: the feature helps average
probability calibration but does not improve, and may harm, the high-confidence
selection tail that matters for FDR control.

Based on this follow-up, the next useful code/setup improvement is not simply
"add PepBERT" globally. A safer path is to expose PLM features behind validation
gates and tune/select them per dataset or per calibration domain, with FDR-tail
metrics as the selection criterion.

## Richer PepBERT Feature Follow-Up

The first full runs used `feature_mode=embedding_summary`, which compresses a
pooled PepBERT embedding into only mean, standard deviation, minimum, and maximum
values. A follow-up mode, `embedding_diagnostics`, was added to test whether more
PLM information helps celegans and immuno2 without changing the backend model.

The diagnostic feature vector includes:

| Feature family | Columns |
| --- | --- |
| Sequence length | `plm_sequence_length`, `plm_log_sequence_length` |
| Compact pooled embedding summary | `plm_embedding_mean`, `plm_embedding_std`, `plm_embedding_min`, `plm_embedding_max` |
| Length-normalized pooled norm | `plm_pooled_l2_norm`, `plm_pooled_l2_norm_per_sqrt_length` |
| Residue-level norm summaries | `plm_residue_norm_mean`, `plm_residue_norm_std`, `plm_residue_norm_min`, `plm_residue_norm_max` |
| Residue-level variation | `plm_residue_embedding_std_mean`, `plm_residue_embedding_std_max` |
| Adjacent residue embedding smoothness | `plm_adjacent_cosine_mean`, `plm_adjacent_cosine_std` |
| Missingness interaction | `plm_missing_x_log_sequence_length` plus `is_missing_peptide_language_model_feature` when `learn_from_missing=true` |

This addresses the low-risk feature-construction options from the follow-up
plan: residue-level summaries, length-normalized norms, sequence length, and
missing-PLM interactions. It does not yet implement PepBERT sequence
plausibility or train-only PCA. PepBERT plausibility is not directly exposed by
the current local PepBERT backend, and PCA needs a fitted train-only feature
transform to avoid leakage into held-out evaluation.

Pre-submit checks:

```bash
uv run pytest tests/calibration/features/test_peptide_language_model.py
uv run python -m py_compile \
  winnow/calibration/features/peptide_language_model.py \
  tests/calibration/features/test_peptide_language_model.py
uv run --extra plm python scripts/plm_manuscript_validation.py \
  --validation-mode in_domain_split \
  --data-dir /tmp/winnow-plm-validation-remote-smoke/data \
  --output-dir /tmp/winnow-plm-validation-remote-smoke/out-indomain-diagnostics-smoke \
  --datasets celegans,immuno2 \
  --max-eval-rows 12 \
  --split-train-fraction 0.5 \
  --plm-feature-mode embedding_diagnostics \
  --plm-batch-size 3
```

The local smoke run completed successfully and wrote outputs to
`/tmp/winnow-plm-validation-remote-smoke/out-indomain-diagnostics-smoke`.
The six-row held-out splits are too small for interpretation; the purpose was
to verify that diagnostic columns run through the calibrator and validation
script before launching the uncapped remote job.

The full uncapped in-domain diagnostic run is configured as:

```bash
HF_HOME=/runs/huggingface \
uv run python scripts/plm_manuscript_validation.py \
  --validation-mode in_domain_split \
  --datasets celegans,immuno2 \
  --max-eval-rows 0 \
  --split-train-fraction 0.5 \
  --plm-feature-mode embedding_diagnostics \
  --plm-batch-size 128 \
  --output-dir /runs/plm-validation-indomain-diagnostics
```

| Setting | Value |
| --- | --- |
| Experiment ID | `414ccd00-71fe-4047-8f9a-f4b2b6d7b5c7` |
| Submission message | `PLM in-domain split validation PepBERT embedding diagnostics` |
| Validation mode | `in_domain_split` |
| Datasets | celegans, immuno2 |
| Split | 50% train, 50% held-out evaluation |
| Row caps | none |
| PLM feature mode | `embedding_diagnostics` |
| Output directory | `/runs/plm-validation-indomain-diagnostics` |
| Status | Succeeded |
| Runtime | 38m07s |

The run completed successfully and wrote outputs to
`/runs/plm-validation-indomain-diagnostics`.

### Diagnostic-Feature Split Sizes

| Dataset | Split | PSMs | Correct PSMs |
| --- | --- | ---: | ---: |
| celegans | train | 116078 | 68425 |
| celegans | held-out eval | 116078 | 68265 |
| immuno2 | train | 10001 | 4012 |
| immuno2 | held-out eval | 10001 | 4152 |

### Diagnostic-Feature Calibration Metrics

| Dataset | Variant | PR-AUC | Brier | ECE |
| --- | --- | ---: | ---: | ---: |
| celegans | confidence only | 0.8864 | 0.1409 | 0.0105 |
| celegans | confidence + PepBERT diagnostics | 0.8856 | 0.1364 | 0.0072 |
| immuno2 | confidence only | 0.8675 | 0.1308 | 0.0872 |
| immuno2 | confidence + PepBERT diagnostics | 0.8590 | 0.1264 | 0.0170 |

### Diagnostic-Feature FDR-Control Results

| Dataset | Variant | Threshold | Accepted PSMs | Realized FDR | Confidence Cutoff | TECE | sTECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| celegans | confidence only | 0.05 | 0 | n/a | n/a | n/a | n/a |
| celegans | confidence only | 0.10 | 39085 | 0.1038 | 0.8218 | 0.0099 | -0.0038 |
| celegans | confidence + PepBERT diagnostics | 0.05 | 7312 | 0.0540 | 0.9379 | 0.0060 | -0.0040 |
| celegans | confidence + PepBERT diagnostics | 0.10 | 43350 | 0.1081 | 0.8139 | 0.0102 | -0.0081 |
| immuno2 | confidence only | 0.05 | 1145 | 0.0638 | 0.8927 | 0.0168 | -0.0138 |
| immuno2 | confidence only | 0.10 | 2099 | 0.1034 | 0.8064 | 0.0184 | -0.0034 |
| immuno2 | confidence + PepBERT diagnostics | 0.05 | 1338 | 0.0703 | 0.9007 | 0.0234 | -0.0203 |
| immuno2 | confidence + PepBERT diagnostics | 0.10 | 2574 | 0.1193 | 0.7825 | 0.0218 | -0.0193 |

### Diagnostic-Feature Interpretation

The richer PepBERT feature set improves average calibration metrics but does not
solve tail FDR control. On celegans, diagnostics improve Brier score and ECE
relative to confidence-only and accept more PSMs at both FDR thresholds, but
realized FDR is above target at 5% and 10%. Compared with the compact
`embedding_summary` in-domain run, diagnostics improve Brier score but reduce
PR-AUC and worsen the 5%/10% tail FDR results.

On immuno2, diagnostics are better than compact summaries for PR-AUC, Brier
score, and ECE, and they accept more PSMs. However, they still reduce PR-AUC
relative to confidence-only and make realized FDR worse at both FDR thresholds.
This keeps the main conclusion unchanged: richer PepBERT embeddings contain
useful calibration signal, but the current calibrator uses that signal in a way
that can over-select in the high-confidence tail.

Runtime also matters. The diagnostic run took 38m07s on the same 8 CPU / 32 GB
RAM Aichor setup, versus 17m58s for the compact in-domain run. Most of the
additional cost came from scoring celegans train and held-out splits with the
larger diagnostic feature mode.
