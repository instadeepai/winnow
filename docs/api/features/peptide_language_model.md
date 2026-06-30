# Peptide Language Model Feature

Adds experimental peptide language model (PLM) features to confidence calibration.

## Purpose

PLM features provide sequence-prior information that is orthogonal to spectrum and chromatography features. They are intended for ablation experiments before production use: a PLM feature can improve calibration on one dataset and hurt tail calibration on another.

## Backends

| Backend | Description | Extra dependencies |
| -------- | ----------- | ------------------ |
| `precomputed` | Reads a scalar PLM score from metadata. | None |
| `pepbert` | Uses PepBERT encoder embeddings and writes compact or diagnostic embedding summaries. | `torch`, `tokenizers`, `huggingface-hub` |
| `esm2` | Uses a Hugging Face ESM-2 model for pseudo-likelihood or embeddings. | `torch`, `transformers` |
| `esmc` | Uses a Hugging Face-compatible ESMC model for pseudo-likelihood or embeddings. | `torch`, `transformers`, model access |

PepBERT is the recommended first backend for peptide-specific experiments. The official PepBERT release exposes encoder embeddings, so use `feature_mode=embedding_summary` for compact local PepBERT features or `feature_mode=embedding_diagnostics` for richer ablation features. Use `precomputed` for externally generated pseudo-likelihood values.

## Columns

For `feature_mode=pseudo_likelihood`:

| Column | Description |
| ------ | ----------- |
| `plm_mean_log_probability` | Length-normalized pseudo-log-likelihood or externally supplied scalar PLM score. |

For `feature_mode=embedding_summary`:

| Column | Description |
| ------ | ----------- |
| `plm_embedding_mean` | Mean of the pooled peptide embedding vector. |
| `plm_embedding_std` | Standard deviation of the pooled peptide embedding vector. |
| `plm_embedding_min` | Minimum value of the pooled peptide embedding vector. |
| `plm_embedding_max` | Maximum value of the pooled peptide embedding vector. |

For `feature_mode=embedding_diagnostics`, Winnow includes the four compact embedding
summary columns above plus:

| Column | Description |
| ------ | ----------- |
| `plm_sequence_length` | Normalized peptide sequence length. |
| `plm_log_sequence_length` | `log1p` normalized peptide sequence length. |
| `plm_pooled_l2_norm` | L2 norm of the pooled peptide embedding. |
| `plm_pooled_l2_norm_per_sqrt_length` | Pooled embedding norm divided by square-root sequence length. |
| `plm_residue_norm_mean` | Mean residue-embedding L2 norm. |
| `plm_residue_norm_std` | Standard deviation of residue-embedding L2 norms. |
| `plm_residue_norm_min` | Minimum residue-embedding L2 norm. |
| `plm_residue_norm_max` | Maximum residue-embedding L2 norm. |
| `plm_residue_embedding_std_mean` | Mean per-dimension standard deviation across residue embeddings. |
| `plm_residue_embedding_std_max` | Maximum per-dimension standard deviation across residue embeddings. |
| `plm_adjacent_cosine_mean` | Mean cosine similarity between adjacent residue embeddings. |
| `plm_adjacent_cosine_std` | Standard deviation of adjacent residue cosine similarities. |
| `plm_missing_x_log_sequence_length` | Missing-PLM indicator multiplied by `plm_log_sequence_length`; non-zero only when `learn_from_missing=true` and the PLM backend cannot score a normalized sequence. |

If `learn_from_missing=true`, `is_missing_peptide_language_model_feature` is also added.

## Usage

Precomputed scalar scores:

```yaml
calibrator:
  features:
    peptide_language_model_feature:
      _target_: winnow.calibration.calibration_features.PeptideLanguageModelFeature
      backend: precomputed
      feature_mode: pseudo_likelihood
      precomputed_column: plm_mean_log_probability
      learn_from_missing: false
```

PepBERT embedding summaries:

```yaml
calibrator:
  features:
    peptide_language_model_feature:
      _target_: winnow.calibration.calibration_features.PeptideLanguageModelFeature
      backend: pepbert
      model_name_or_path: dzjxzyd/PepBERT-large-UniParc
      feature_mode: embedding_summary
      batch_size: 16
      device: null
      cache_dir: null
      learn_from_missing: false
```

PepBERT embedding diagnostics:

```yaml
calibrator:
  features:
    peptide_language_model_feature:
      _target_: winnow.calibration.calibration_features.PeptideLanguageModelFeature
      backend: pepbert
      model_name_or_path: dzjxzyd/PepBERT-large-UniParc
      feature_mode: embedding_diagnostics
      batch_size: 128
      device: null
      cache_dir: null
      learn_from_missing: true
```

ESM-2 pseudo-likelihood:

```yaml
calibrator:
  features:
    peptide_language_model_feature:
      _target_: winnow.calibration.calibration_features.PeptideLanguageModelFeature
      backend: esm2
      model_name_or_path: facebook/esm2_t6_8M_UR50D
      feature_mode: pseudo_likelihood
      batch_size: 16
      learn_from_missing: false
```

## Notes

- PLM dependencies are optional and loaded lazily.
- Predicted peptide tokens are normalized by stripping PTM annotations to base amino acids before scoring.
- Invalid or empty normalized peptides are filtered by default. Set `learn_from_missing=true` to retain them with imputed zero values plus a missingness indicator.
- PLM outputs are cached by normalized peptide sequence within each run.
- Models trained with PLM features are not compatible with calibrator checkpoints trained without those feature columns.
