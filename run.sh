#!/bin/sh

# Default values
MODEL_DIR=${1:-model}
DATA_DIR=${2:-data}
TRAIN_CONFIG=${3:-configs/train.yaml}
PREDICT_LABELLED_CONFIG=${4:-configs/predict_labelled.yaml}
PREDICT_DE_NOVO_CONFIG=${5:-configs/predict_de_novo.yaml}

# Train
winnow train \
    --data-source=winnow \
    --dataset-config-path=${TRAIN_CONFIG} \
    --model-output-folder=${MODEL_DIR} \
    --dataset-output-path=${DATA_DIR}/train_output.csv

# Evaluate on labelled data using winnow method
winnow predict \
    --data-source=winnow \
    --dataset-config-path=${PREDICT_LABELLED_CONFIG} \
    --model-folder=${MODEL_DIR} \
    --method=winnow \
    --fdr-threshold=0.05 \
    --confidence-column="calibrated_confidence" \
    --output-path=${DATA_DIR}/labelled_winnow_predict_output.csv

# Evaluate on labelled data using database-ground method
winnow predict \
    --data-source=winnow \
    --dataset-config-path=${PREDICT_LABELLED_CONFIG} \
    --model-folder=${MODEL_DIR} \
    --method=database-ground \
    --fdr-threshold=0.05 \
    --confidence-column="calibrated_confidence" \
    --output-path=${DATA_DIR}/labelled_database_ground_predict_output.csv

# Evaluate on de novo data using winnow method
winnow predict \
    --data-source=instanovo \
    --dataset-config-path=${PREDICT_DE_NOVO_CONFIG} \
    --model-folder=${MODEL_DIR} \
    --method=winnow \
    --fdr-threshold=0.05 \
    --confidence-column="calibrated_confidence" \
    --output-path=${DATA_DIR}/de_novo_winnow_predict_output.csv
