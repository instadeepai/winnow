# This Makefile provides shortcut commands to facilitate local development.

# Common variables
PACKAGE_NAME = winnow

# Train variables
NUM_NODES = 1
BATCH_SIZE = 12

# LAST_COMMIT returns the current HEAD commit
LAST_COMMIT = $(shell git rev-parse --short HEAD)

# VERSION represents a clear statement of which tag based version of the repository you're actually running.
# If you run a tag based version, it returns the according HEAD tag, otherwise it returns:
# * `LAST_COMMIT-staging` if no tags exist
# * `BASED_TAG-SHORT_SHA_COMMIT-staging` if a previous tag exist
VERSION := $(shell git describe --always --exact-match --abbrev=0 --tags $(LAST_COMMIT) 2> /dev/null)
ifndef VERSION
	BASED_VERSION := $(shell git describe --always --abbrev=3 --tags $(git rev-list --tags --max-count=1))
	ifndef BASED_VERSION
	VERSION = $(LAST_COMMIT)-staging
	else
	VERSION = $(BASED_VERSION)-staging
	endif
endif

# Docker variables
DOCKER_HOME_DIRECTORY = "/app"
DOCKER_RUNS_DIRECTORY = "/runs"

DOCKERFILE := Dockerfile

DOCKER_IMAGE_NAME = winnow
DOCKER_IMAGE_TAG = $(VERSION)
DOCKER_IMAGE = $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)

DOCKER_RUN_FLAGS = --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --shm-size='1gb'
DOCKER_RUN_FLAGS_VOLUME_MOUNT_HOME = $(DOCKER_RUN_FLAGS) --volume $(PWD):$(DOCKER_HOME_DIRECTORY)
DOCKER_RUN_FLAGS_VOLUME_MOUNT_RUNS = $(DOCKER_RUN_FLAGS) --volume $(PWD)/runs:$(DOCKER_RUNS_DIRECTORY)
DOCKER_RUN = docker run $(DOCKER_RUN_FLAGS) $(IMAGE_NAME)

PYTEST = pytest --alluredir=allure_results --cov-report=html --cov --cov-config=.coveragerc --random-order --verbose .
COVERAGE = coverage report -m

#################################################################################
## Docker build commands																#
#################################################################################

.PHONY: build build-arm

define docker_buildx_template
	docker buildx build --platform=$(1) --progress=plain . \
		-f $(DOCKERFILE) -t $(2) --build-arg GID=$(shell id -g) \
		--build-arg UID=$(shell id -u)  --build-arg LAST_COMMIT=$(LAST_COMMIT) \
		--build-arg VERSION=$(VERSION) --build-arg HOME_DIRECTORY=$(DOCKER_HOME_DIRECTORY) \
		--build-arg RUNS_DIRECTORY=$(DOCKER_RUNS_DIRECTORY)
endef

## Build Docker image for winnow on AMD64
build:
	$(call docker_buildx_template,linux/amd64,$(DOCKER_IMAGE))

## Build Docker image for winnow on ARM64
build-arm:
	$(call docker_buildx_template,linux/arm64,$(DOCKER_IMAGE))

#################################################################################
## Install packages commands													#
#################################################################################

.PHONY: compile install install-dev install-all

## Compile all the pinned requirements*.txt files from the unpinned requirements*.in files
compile:
	pip install --upgrade uv
	rm -f *.txt
	uv pip compile requirements.in --emit-index-url  --output-file=requirements.txt

## Install required packages
install:
	pip install --upgrade uv
	uv pip install -r requirements.txt

##  Sync pinned dependencies with your virtual environment
sync:
	pip install --upgrade uv
	uv pip sync requirements.txt

#################################################################################
## Development commands														 	#
#################################################################################

.PHONY: tests coverage test-docker coverage-docker bash set-gcp-credentials set-ceph-credentials

## Run all tests
tests:
	python -m winnow.scripts.get_zenodo_record
	$(PYTEST)

## Calculate the code coverage
coverage:
	$(COVERAGE)

## Run all tests in the Docker Image
test-docker:
	docker run $(DOCKER_RUN_FLAGS) $(DOCKER_IMAGE) nvidia-smi && $(PYTEST)

## Calculate the code coverage in the Docker image
coverage-docker:
	docker run $(DOCKER_RUN_FLAGS) $(DOCKER_IMAGE) nvidia-smi && $(PYTEST) && $(COVERAGE)

## Open a bash shell in the Docker image
bash:
	docker run -it $(DOCKER_RUN_FLAGS) $(DOCKER_IMAGE) /bin/bash

## Set the GCP credentials
set-gcp-credentials:
	python scripts/set_gcp_credentials.py
	gcloud auth activate-service-account dtu-denovo-sa@ext-dtu-denovo-sequencing-gcp.iam.gserviceaccount.com --key-file=ext-dtu-denovo-sequencing-gcp.json --project=ext-dtu-denovo-sequencing-gcp

## Set the Ceph credentials
set-ceph-credentials:
	python scripts/set_ceph_credentials.py

#################################################################################
## Analysis commands															#
#################################################################################

.PHONY: train-general-model evaluate-general-model-validation-datasets evaluate-general-model-external-datasets add-db-fdr-validation-datasets add-db-fdr-external-datasets evaluate-calibrator-generalisation analyze-features evaluate-holdout-sets

## Analyze the feature importance and correlations
analyze-features: set-ceph-credentials set-gcp-credentials
	# Copy the data from Ceph bucket to the local data directory
	mkdir -p data
	aws s3 cp s3://winnow-g88rh/validation_datasets_corrected/spectrum_data/labelled/train_spectrum_all_datasets.parquet data/ --profile winnow
	aws s3 cp s3://winnow-g88rh/validation_datasets_corrected/beam_preds/labelled/train_beam_all_datasets.csv data/ --profile winnow
	python scripts/analyze_features.py --train-spectrum-path data/train_spectrum_all_datasets.parquet --train-predictions-path data/train_beam_all_datasets.csv --output-dir mlp_feature_importance_results/ --n-background-samples 500
	# Copy the results back to Ceph bucket
	aws s3 cp mlp_feature_importance_results s3://winnow-g88rh/classifier_comparison/ --recursive --profile winnow
	# Copy the results back to Google Cloud Storage
	gsutil -m cp -R mlp_feature_importance_results/ gs://winnow-fdr/classifier_comparison/dt_feature_importance_results/

## Assess calibrator generalisation
calibrator-generalisation: set-ceph-credentials set-gcp-credentials
	mkdir -p data/spectrum_data
	mkdir -p data/beam_preds
	mkdir -p models
	mkdir -p results
	# Copy the data from Ceph bucket to the local data directory
	aws s3 cp s3://winnow-g88rh/validation_datasets_corrected/spectrum_data/labelled/ data/spectrum_data/ --recursive --exclude "*" --include "dataset*" --profile winnow --endpoint-url https://s3.aichor-dataplane-prod.eqx.ext.id-baremetal.net
	aws s3 cp s3://winnow-g88rh/validation_datasets_corrected/beam_preds/labelled/ data/beam_preds/ --recursive --exclude "*" --include "*-annotated_beam_preds.csv" --profile winnow --endpoint-url https://s3.aichor-dataplane-prod.eqx.ext.id-baremetal.net
	# Run the generalisation evaluation script
	python scripts/evaluate_calibrator_generalisation.py \
		--data-source instanovo \
		--config-dir configs/calibrator_generalisation \
		--model-output-dir models/generalisation \
		--results-output-dir results/generalisation
	# Copy the results back to Ceph bucket
	aws s3 cp results/ s3://winnow-g88rh/calibrator_generalisation/ --recursive --profile winnow
	aws s3 cp models/ s3://winnow-g88rh/calibrator_generalisation/ --recursive --profile winnow
	# Copy the results back to Google Cloud Storage
	gsutil -m cp -R results/ gs://winnow-fdr/calibrator_generalisation/
	gsutil -m cp -R models/ gs://winnow-fdr/calibrator_generalisation/

val_datasets := gluc helaqc herceptin immuno sbrodae snakevenoms woundfluids
ext_datasets := PXD014877 PXD019483 PXD023064

## Train general model
train-general-model: set-ceph-credentials set-gcp-credentials
	# Make folders
	mkdir -p validation_datasets_corrected/beam_preds/labelled
	mkdir -p validation_datasets_corrected/spectrum_data/labelled
	# Copy the data from Ceph bucket to the local data directory
	aws s3 cp s3://winnow-g88rh/validation_datasets_corrected/beam_preds/labelled/train_beam_all_datasets.csv validation_datasets_corrected/beam_preds/labelled/ --profile winnow
	aws s3 cp s3://winnow-g88rh/validation_datasets_corrected/spectrum_data/labelled/train_spectrum_all_datasets.parquet validation_datasets_corrected/spectrum_data/labelled/ --profile winnow
	# Run the generalisation evaluation script
	winnow train --data-source instanovo --dataset-config-path configs/validation_data/train_general_model.yaml --model-output-folder general_model/model --dataset-output-path general_model/results/general_model_training_results.csv
	# Copy the results back to Ceph bucket
	aws s3 cp general_model/ s3://winnow-g88rh/general_model/ --recursive --profile winnow

## Evaluate general model on validation datasets
evaluate-general-model-validation-datasets: set-ceph-credentials set-gcp-credentials
	# Make folders
	mkdir -p validation_datasets_corrected/winnow_metadata/labelled
	mkdir -p validation_datasets_corrected/winnow_metadata/de_novo
	mkdir -p validation_datasets_corrected/winnow_metadata/raw
	mkdir -p validation_datasets_corrected/beam_preds
	mkdir -p validation_datasets_corrected/spectrum_data
	mkdir -p general_model/model
	mkdir -p fasta
	# Copy the data from Ceph bucket to the local data directory
	aws s3 cp s3://winnow-g88rh/validation_datasets_corrected/beam_preds/ validation_datasets_corrected/beam_preds/ --recursive --profile winnow
	aws s3 cp s3://winnow-g88rh/validation_datasets_corrected/spectrum_data/ validation_datasets_corrected/spectrum_data/ --recursive --profile winnow
	aws s3 cp s3://winnow-g88rh/general_model/model/ general_model/model/ --recursive --profile winnow
	aws s3 cp s3://winnow-g88rh/fasta/ fasta/ --recursive --profile winnow
	# Run the generalisation evaluation script
	# 1. Evaluate labelled data
	winnow predict --data-source instanovo --dataset-config-path configs/validation_data/test_general_model_validation_data.yaml --model-folder general_model/model --method winnow --fdr-threshold 0.05 --confidence-column calibrated_confidence --output-path validation_datasets_corrected/winnow_metadata/labelled/general_test_winnow_output.csv --no-label-shift --confidence-cutoff-path validation_datasets_corrected/winnow_metadata/labelled/general_test_winnow_conf_cutoff.txt
	python scripts/add_db_fdr.py --input-path validation_datasets_corrected/winnow_metadata/labelled/general_test_winnow_output.csv --output-path validation_datasets_corrected/winnow_metadata/labelled/general_test_winnow_output.csv --confidence-column calibrated_confidence --confidence-cutoff-path validation_datasets_corrected/winnow_metadata/labelled/general_test_winnow_output_conf_cutoff.txt
	python scripts/add_db_fdr.py --input-path validation_datasets_corrected/winnow_metadata/labelled/general_test_winnow_output.csv --output-path validation_datasets_corrected/winnow_metadata/labelled/general_test_winnow_output_raw_conf.csv --confidence-column confidence --confidence-cutoff-path validation_datasets_corrected/winnow_metadata/labelled/general_test_winnow_output_raw_conf_cutoff.txt
	# 2. Evaluate unlabelled data
	@for ds in $(val_datasets); do \
		winnow predict --data-source instanovo --dataset-config-path configs/validation_data/$${ds}_de_novo.yaml --model-folder general_model/model --method winnow --fdr-threshold 0.05 --confidence-column calibrated_confidence --output-path validation_datasets_corrected/winnow_metadata/de_novo/$${ds}_de_novo_preds.csv --no-label-shift --confidence-cutoff-path validation_datasets_corrected/winnow_metadata/de_novo/$${ds}_de_novo_conf_cutoff.txt; \
	done
	# 3. Evaluate full search space
	@for ds in $(val_datasets); do \
		winnow predict --data-source instanovo --dataset-config-path configs/validation_data/$${ds}_raw.yaml --model-folder general_model/model --method winnow --fdr-threshold 0.05 --confidence-column calibrated_confidence --output-path validation_datasets_corrected/winnow_metadata/raw/$${ds}_raw_preds.csv --no-label-shift --confidence-cutoff-path validation_datasets_corrected/winnow_metadata/raw/$${ds}_raw_conf_cutoff.txt; \
	done
	# Map peptides to proteomes
	# gluc
	python scripts/map_peptides_to_proteomes.py --metadata-csv validation_datasets_corrected/winnow_metadata/raw/gluc_raw_preds.csv --fasta-file fasta/human.fasta --output-csv validation_datasets_corrected/winnow_metadata/raw/gluc_raw_preds.csv
	# helaqc
	python scripts/map_peptides_to_proteomes.py --metadata-csv validation_datasets_corrected/winnow_metadata/raw/helaqc_raw_preds.csv --fasta-file fasta/human.fasta --output-csv validation_datasets_corrected/winnow_metadata/raw/helaqc_raw_preds.csv
	# herceptin
	python scripts/map_peptides_to_proteomes.py --metadata-csv validation_datasets_corrected/winnow_metadata/raw/herceptin_raw_preds.csv --fasta-file fasta/herceptin.fasta --output-csv validation_datasets_corrected/winnow_metadata/raw/herceptin_raw_preds.csv
	# sbrodae
	python scripts/map_peptides_to_proteomes.py --metadata-csv validation_datasets_corrected/winnow_metadata/raw/sbrodae_raw_preds.csv --fasta-file fasta/Sb_proteome.fasta --output-csv validation_datasets_corrected/winnow_metadata/raw/sbrodae_raw_preds.csv
	# snakevenoms
	python scripts/map_peptides_to_proteomes.py --metadata-csv validation_datasets_corrected/winnow_metadata/raw/snakevenoms_raw_preds.csv --fasta-file fasta/uniprot-serpentes-2022.05.09.fasta --output-csv validation_datasets_corrected/winnow_metadata/raw/snakevenoms_raw_preds.csv
	# NB. we ignore immuno and woundfluids as contains many species
	# Copy the results back to Ceph bucket
	aws s3 cp validation_datasets_corrected/winnow_metadata/labelled/general_test_winnow_output.csv s3://winnow-g88rh/validation_datasets_corrected/winnow_metadata/labelled/ --profile winnow
	aws s3 cp validation_datasets_corrected/winnow_metadata/de_novo/ s3://winnow-g88rh/validation_datasets_corrected/winnow_metadata/de_novo/ --recursive --profile winnow
	# Copy to Google Cloud Storage
	gsutil -m cp -R validation_datasets_corrected/winnow_metadata/ gs://winnow-fdr/validation_datasets_corrected/winnow_metadata/

## Evaluate general model on external datasets
evaluate-general-model-external-datasets: set-ceph-credentials set-gcp-credentials
	# Make folders
	mkdir -p external_datasets/winnow_metadata/lcfm
	mkdir -p external_datasets/winnow_metadata/de_novo
	mkdir -p external_datasets/winnow_metadata/acfm
	mkdir -p external_datasets/beam_preds
	mkdir -p external_datasets/spectrum_data
	mkdir -p general_model/model
	mkdir -p fasta
	# Copy the data from Ceph bucket to the local data directory
	aws s3 cp s3://winnow-g88rh/external_datasets/spectrum_data/ external_datasets/spectrum_data/ --recursive --profile winnow
	aws s3 cp s3://winnow-g88rh/external_datasets/beam_preds/ external_datasets/beam_preds/ --recursive --profile winnow
	aws s3 cp s3://winnow-g88rh/general_model/model/ general_model/model/ --recursive --profile winnow
	aws s3 cp s3://winnow-g88rh/fasta/ fasta/ --recursive --profile winnow
	# Run the generalisation evaluation script
	# 1. Evaluate labelled data
	@for ds in $(ext_datasets); do \
		winnow predict --data-source instanovo --dataset-config-path configs/external_datasets/$${ds}_lcfm.yaml --model-folder general_model/model --method winnow --fdr-threshold 0.05 --confidence-column calibrated_confidence --output-path external_datasets/winnow_metadata/lcfm/$${ds}_lcfm_preds_winnow.csv --no-label-shift --confidence-cutoff-path external_datasets/winnow_metadata/lcfm/$${ds}_lcfm_conf_cutoff_winnow.txt; \
		python scripts/add_db_fdr.py --input-path external_datasets/winnow_metadata/lcfm/$${ds}_lcfm_preds_winnow.csv --output-path external_datasets/winnow_metadata/lcfm/$${ds}_lcfm_preds.csv --confidence-column calibrated_confidence --confidence-cutoff-path external_datasets/winnow_metadata/lcfm/$${ds}_lcfm_conf_cutoff_dbg.txt; \
		python scripts/add_db_fdr.py --input-path external_datasets/winnow_metadata/lcfm/$${ds}_lcfm_preds_winnow.csv --output-path external_datasets/winnow_metadata/lcfm/$${ds}_lcfm_preds_raw_conf.csv --confidence-column confidence --confidence-cutoff-path external_datasets/winnow_metadata/lcfm/$${ds}_lcfm_conf_cutoff_dbg_raw_conf.txt; \
	done
	# 2. Evaluate de novo unlabelled data
	@for ds in $(ext_datasets); do \
		winnow predict --data-source instanovo --dataset-config-path configs/external_datasets/$${ds}_de_novo.yaml --model-folder general_model/model --method winnow --fdr-threshold 0.05 --confidence-column calibrated_confidence --output-path external_datasets/winnow_metadata/de_novo/$${ds}_de_novo_preds.csv --no-label-shift --confidence-cutoff-path external_datasets/winnow_metadata/de_novo/$${ds}_de_novo_conf_cutoff_winnow.txt; \
	done
	# 3. Evaluate raw unlabelled data
	@for ds in $(ext_datasets); do \
		winnow predict --data-source instanovo --dataset-config-path configs/external_datasets/$${ds}_acfm.yaml --model-folder general_model/model --method winnow --fdr-threshold 0.05 --confidence-column calibrated_confidence --output-path external_datasets/winnow_metadata/acfm/$${ds}_acfm_preds.csv --no-label-shift --confidence-cutoff-path external_datasets/winnow_metadata/acfm/$${ds}_acfm_conf_cutoff_winnow.txt; \
	done
	# Compute proteome mapping
	python scripts/map_peptides_to_proteomes.py --metadata-csv external_datasets/winnow_metadata/acfm/PXD014877_acfm_preds.csv --fasta-file fasta/Celegans.fasta --output-csv external_datasets/winnow_metadata/acfm/PXD014877_acfm_preds.csv
	python scripts/map_peptides_to_proteomes.py --metadata-csv external_datasets/winnow_metadata/acfm/PXD019483_acfm_preds.csv --fasta-file fasta/human.fasta --output-csv external_datasets/winnow_metadata/acfm/PXD019483_acfm_preds.csv
	python scripts/map_peptides_to_proteomes.py --metadata-csv external_datasets/winnow_metadata/acfm/PXD023064_acfm_preds.csv --fasta-file fasta/human.fasta --output-csv external_datasets/winnow_metadata/acfm/PXD023064_acfm_preds.csv
	# Copy the results back to Ceph bucket
	aws s3 cp external_datasets/winnow_metadata/ s3://winnow-g88rh/external_datasets/winnow_metadata/ --recursive --profile winnow
	# Copy to Google Cloud Storage
	gsutil -m cp -R external_datasets/winnow_metadata/ gs://winnow-fdr/external_datasets/winnow_metadata
	gsutil -m cp -R general_model/model/ gs://winnow-fdr/general_model/model

## Add database-grounded FDR to validation datasets
add-db-fdr-validation-datasets: set-ceph-credentials set-gcp-credentials
	# Make folders
	mkdir -p validation_datasets_corrected/winnow_metadata/labelled
	# Copy the data from Ceph bucket to the local data directory
	aws s3 cp s3://winnow-g88rh/validation_datasets_corrected/winnow_metadata/labelled/general_test_winnow_output.csv validation_datasets_corrected/winnow_metadata/labelled/ --profile winnow
	# Add database-grounded FDR to labelled data
	python scripts/add_db_fdr.py \
		--input-path validation_datasets_corrected/winnow_metadata/labelled/general_test_winnow_output.csv \
		--output-path validation_datasets_corrected/winnow_metadata/labelled/general_test.csv \
		--confidence-column calibrated_confidence
	# Copy the results back to Ceph bucket
	aws s3 cp validation_datasets_corrected/winnow_metadata/labelled/general_test.csv s3://winnow-g88rh/validation_datasets_corrected/winnow_metadata/labelled/ --profile winnow
	# Copy to Google Cloud Storage
	gsutil -m cp -R validation_datasets_corrected/winnow_metadata/labelled/general_test.csv gs://winnow-fdr/validation_datasets_corrected/winnow_metadata/labelled/

## Add database-grounded FDR to external datasets
add-db-fdr-external-datasets: # set-ceph-credentials set-gcp-credentials
	# # Make folders
	# mkdir -p external_datasets/winnow_metadata/lcfm
	# # Copy the data from Ceph bucket to the local data directory
	# aws s3 cp s3://winnow-g88rh/external_datasets/winnow_metadata/lcfm/ external_datasets/winnow_metadata/lcfm/ --recursive --profile winnow
	# Add database-grounded FDR to labelled data (LCFM)
	@for ds in $(ext_datasets); do \
		python scripts/add_db_fdr.py \
			--input-path external_datasets/winnow_metadata/lcfm/$${ds}_lcfm_preds_winnow.csv \
			--output-path external_datasets/winnow_metadata/lcfm/$${ds}_lcfm_preds.csv \
			--confidence-column calibrated_confidence; \
	done
	# # Copy the results back to Ceph bucket
	# aws s3 cp external_datasets/winnow_metadata/lcfm/ s3://winnow-g88rh/external_datasets/winnow_metadata/lcfm/ --recursive --profile winnow
	# # Copy to Google Cloud Storage
	# gsutil -m cp -R external_datasets/winnow_metadata/lcfm/ gs://winnow-fdr/external_datasets/winnow_metadata/lcfm/

## Evaluate combined data
evaluate-combined-data:
	# Train model on combined data
	winnow train --data-source instanovo --dataset-config-path configs/combined_data/val.yaml --model-output-folder general_model/model --dataset-output-path general_model/results/general_model_training_results.csv
	# Validation hold-out set
	winnow predict --data-source instanovo --dataset-config-path configs/validation_data/test_general_model_validation_data.yaml --model-folder general_model/model --method winnow --fdr-threshold 0.05 --confidence-column "calibrated_confidence" --output-path validation_datasets_corrected/winnow_metadata/labelled/general_test_winnow_output.csv
	python scripts/add_db_fdr.py \
		--input-path validation_datasets_corrected/winnow_metadata/labelled/general_test_winnow_output.csv \
		--output-path validation_datasets_corrected/winnow_metadata/labelled/general_test.csv \
		--confidence-column calibrated_confidence
	# External datasets
	@for ds in $(ext_datasets); do \
		winnow predict --data-source instanovo --dataset-config-path configs/external_datasets/$${ds}_lcfm.yaml --model-folder general_model/model --method winnow --fdr-threshold 0.05 --confidence-column "calibrated_confidence" --output-path external_datasets/winnow_metadata/lcfm/$${ds}_lcfm_preds.csv; \
	done
	@for ds in $(ext_datasets); do \
		python scripts/add_db_fdr.py \
			--input-path external_datasets/winnow_metadata/lcfm/$${ds}_lcfm_preds_winnow.csv \
			--output-path external_datasets/winnow_metadata/lcfm/$${ds}_lcfm_preds.csv \
			--confidence-column calibrated_confidence; \
	done
	# Calibrator generalisation
	python scripts/evaluate_calibrator_generalisation.py \
		--data-source instanovo \
		--config-dir configs/calibrator_generalisation \
		--model-output-dir models/generalisation \
		--results-output-dir results/generalisation
# NB, ignore the PXD014877 results as it was part of the training set.

## Evaluate holdout sets
evaluate-holdout-sets: set-ceph-credentials set-gcp-credentials
	# Copy data from Ceph bucket to local directory
	aws s3 cp s3://winnow-g88rh/validation_datasets_corrected/spectrum_data/ validation_datasets_corrected/spectrum_data/ --recursive --profile winnow
	aws s3 cp s3://winnow-g88rh/validation_datasets_corrected/beam_preds/ validation_datasets_corrected/beam_preds/ --recursive --profile winnow
	aws s3 cp s3://winnow-g88rh/fasta/ fasta/ --recursive --profile winnow
	# Make folders
	mkdir -p holdout_models
	mkdir -p holdout_sets
	mkdir -p holdout_results/plots
	# Create holdout sets
	python scripts/create_holdout_sets.py
	# Train models and evaluate holdout sets
	@for holdout_set in $(val_datasets); do \
		echo "Evaluating holdout set: $$holdout_set"; \
		mkdir -p holdout_models/all_less_$${holdout_set}; \
		winnow train --data-source instanovo --dataset-config-path configs/holdout/all_less_$${holdout_set}.yaml --model-output-folder holdout_models/all_less_$${holdout_set} --dataset-output-path holdout_results/all_less_$${holdout_set}_train_results.csv; \
		winnow predict --data-source instanovo --dataset-config-path configs/validation_data/$${holdout_set}_labelled.yaml --model-folder holdout_models/all_less_$${holdout_set} --method winnow --fdr-threshold  --confidence-column "calibrated_confidence" --output-path holdout_results/all_less_$${holdout_set}_labelled_test_results.csv --confidence-cutoff-path holdout_results/all_less_$${holdout_set}_winnow_labelled_confidence_cutoff.txt; \
		python scripts/add_db_fdr.py --input-path holdout_results/all_less_$${holdout_set}_labelled_test_results.csv --output-path holdout_results/all_less_$${holdout_set}_labelled_test_results_with_db_fdr.csv --confidence-column calibrated_confidence --fdr-threshold  --confidence-cutoff-path holdout_results/all_less_$${holdout_set}_dbg_labelled_confidence_cutoff.txt; \
		winnow predict --data-source instanovo --dataset-config-path configs/validation_data/$${holdout_set}_raw.yaml --model-folder holdout_models/all_less_$${holdout_set} --method winnow --fdr-threshold  --confidence-column "calibrated_confidence" --output-path holdout_results/all_less_$${holdout_set}_raw_test_results.csv --confidence-cutoff-path holdout_results/all_less_$${holdout_set}_winnow_raw_confidence_cutoff.txt; \
	done
	# Map peptides to proteomes
	# gluc
	python scripts/map_peptides_to_proteomes.py --metadata-csv holdout_results/all_less_gluc_raw_test_results.csv --fasta-file fasta/human.fasta --output-csv holdout_results/all_less_gluc_raw_test_results.csv
	# helaqc
	python scripts/map_peptides_to_proteomes.py --metadata-csv holdout_results/all_less_helaqc_raw_test_results.csv --fasta-file fasta/human.fasta --output-csv holdout_results/all_less_helaqc_raw_test_results.csv
	# herceptin
	python scripts/map_peptides_to_proteomes.py --metadata-csv holdout_results/all_less_herceptin_raw_test_results.csv --fasta-file fasta/herceptin.fasta --output-csv holdout_results/all_less_herceptin_raw_test_results.csv
	# sbrodae
	python scripts/map_peptides_to_proteomes.py --metadata-csv holdout_results/all_less_sbrodae_raw_test_results.csv --fasta-file fasta/Sb_proteome.fasta --output-csv holdout_results/all_less_sbrodae_raw_test_results.csv
	# snakevenoms
	python scripts/map_peptides_to_proteomes.py --metadata-csv holdout_results/all_less_snakevenoms_raw_test_results.csv --fasta-file fasta/uniprot-serpentes-2022.05.09.fasta --output-csv holdout_results/all_less_snakevenoms_raw_test_results.csv
	# NB. we ignore immuno and woundfluids as contains many species
	# Create plots
	python scripts/create_holdout_plots.py
	# Copy results to Ceph bucket
	aws s3 cp holdout_results/ s3://winnow-g88rh/poster/holdout_results/ --recursive --profile winnow
	aws s3 cp holdout_models/ s3://winnow-g88rh/poster/holdout_models/ --recursive --profile winnow
	# Copy results to Google Cloud Storage
	gsutil -m cp -R holdout_results/ gs://winnow-fdr/poster/holdout_results/
	gsutil -m cp -R holdout_models/ gs://winnow-fdr/poster/holdout_models/
