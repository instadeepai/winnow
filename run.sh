#!/bin/sh

winnow train --data-source=instanovo --dataset-config-path=configs/train.yaml --model-output-folder=model --dataset-output-path=data/train_output.csv

winnow predict --data-source=instanovo --dataset-config-path=configs/predict.yaml --model-folder=model --method=winnow --fdr-threshold=0.05 --confidence-column="calibrated_confidence" --output-path=data/predict_output.csv
