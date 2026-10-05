#!/usr/bin/env bash
# Train LOO generalisation models, plot heatmaps, upload to S3, then compare to Figshare.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

S3_PREFIX="${S3_PREFIX:-s3://winnow-g88rh/revisions/new_loo_run}"
RESULTS_DIR="${RESULTS_DIR:-paper_results/generalisation_resplit_es}"
PLOTS_DIR="${PLOTS_DIR:-paper_plots/generalisation_resplit_es}"
FIGSHARE_REF="${FIGSHARE_REF:-paper_data/generalisation_figshare_deposit}"
TRAIN_PARQUET="${TRAIN_PARQUET:-paper_data/winnow-ms-datasets/general_model_training_set/train.parquet}"
TRAIN_PREDS="${TRAIN_PREDS:-paper_data/winnow-ms-datasets/general_model_training_set/train_preds.csv}"
FIGSHARE_ARTICLE_ID="${FIGSHARE_ARTICLE_ID:-30147601}"
FIGSHARE_VERSION="${FIGSHARE_VERSION:-8}"

RESULTS_CSV="${RESULTS_DIR}/calibrator_generalisation_results.csv"
COMPARE_LOG="${RESULTS_DIR}/compare_to_figshare.log"

mkdir -p "$RESULTS_DIR" "$PLOTS_DIR"
LOG_FILE="${RESULTS_DIR}/pipeline.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== Winnow LOO generalisation pipeline ==="
echo "repo:          $REPO_ROOT"
echo "results:       $RESULTS_DIR"
echo "plots:         $PLOTS_DIR"
echo "s3:            $S3_PREFIX"
echo "started:       $(date -Is)"

echo "=== Phase 0: download inputs (HF general_model_training_set via Makefile.paper) ==="
make -f Makefile.paper download-paper-datasets

for path in "$TRAIN_PARQUET" "$TRAIN_PREDS"; do
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing training input after download: $path" >&2
    exit 1
  fi
done

echo "=== Phase 1: train + evaluate (early stopping, OOD feature cache) ==="
uv run python paper_scripts/evaluate_calibrator_generalisation.py \
  --train-parquet "$TRAIN_PARQUET" \
  --train-predictions "$TRAIN_PREDS" \
  --model-output-dir "${RESULTS_DIR}/models" \
  --results-output-dir "$RESULTS_DIR"

if [[ ! -f "$RESULTS_CSV" ]]; then
  echo "ERROR: expected results CSV missing: $RESULTS_CSV" >&2
  exit 1
fi

echo "=== Phase 2: PR-AUC heatmaps ==="
uv run python paper_scripts/plot_calibrator_generalisation_heatmap.py \
  --results-path "$RESULTS_CSV" \
  --plots-dir "$PLOTS_DIR"

echo "=== Phase 3: upload artefacts to S3 ==="
aws s3 sync "$RESULTS_DIR" "${S3_PREFIX}/results/" --only-show-errors
aws s3 sync "$PLOTS_DIR" "${S3_PREFIX}/plots/" --only-show-errors
aws s3 cp "$LOG_FILE" "${S3_PREFIX}/pipeline.log"

echo "=== Phase 4: download Figshare reference CSV ==="
mkdir -p "$FIGSHARE_REF"
uv run python paper_scripts/download_figshare_article.py \
  --article-id "$FIGSHARE_ARTICLE_ID" \
  --version "$FIGSHARE_VERSION" \
  --output-dir "$FIGSHARE_REF" \
  --include "generalisation/calibrator_generalisation_results.csv"

DEPOSIT_CSV="$(find "$FIGSHARE_REF" -name calibrator_generalisation_results.csv -type f | head -n 1)"
if [[ -z "$DEPOSIT_CSV" || ! -f "$DEPOSIT_CSV" ]]; then
  echo "ERROR: Figshare calibrator_generalisation_results.csv not found under $FIGSHARE_REF" >&2
  exit 1
fi

echo "=== Phase 5: compare recomputed vs Figshare (logged) ==="
uv run python paper_scripts/compare_generalisation_heatmaps.py \
  "$DEPOSIT_CSV" \
  "$RESULTS_CSV" | tee "$COMPARE_LOG"

aws s3 cp "$COMPARE_LOG" "${S3_PREFIX}/compare_to_figshare.log"

echo "=== Done ==="
echo "finished:      $(date -Is)"
echo "s3 results:    ${S3_PREFIX}/results/"
echo "s3 plots:      ${S3_PREFIX}/plots/"
echo "compare log:   ${S3_PREFIX}/compare_to_figshare.log"
