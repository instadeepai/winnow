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

.PHONY: train-general-model evaluate-general-model-validation-datasets evaluate-general-model-external-datasets add-db-fdr-validation-datasets add-db-fdr-external-datasets

## Analyze the feature importance and correlations
analyze-features: set-ceph-credentials set-gcp-credentials
	# Copy the data from Ceph bucket to the local data directory
	mkdir -p data
	aws s3 cp s3://winnow-g88rh/validation_datasets_corrected/spectrum_data/labelled/train_spectrum_all_datasets.parquet data/ --profile winnow
	aws s3 cp s3://winnow-g88rh/validation_datasets_corrected/beam_preds/labelled/train_beam_all_datasets.csv data/ --profile winnow
	python scripts/analyze_features.py --train-spectrum-path data/train_spectrum_all_datasets.parquet --train-predictions-path data/train_beam_all_datasets.csv --output-dir mlp_feature_importance_results/ --n-background-samples 500
	# Copy the results back to Ceph bucket
	# aws s3 cp results/ s3://winnow-g88rh/classifier_comparison/ --recursive --profile winnow
	# Copy the results back to Google Cloud Storage
	gsutil -m cp -R mlp_feature_importance_results/ gs://winnow-fdr/classifier_comparison/dt_feature_importance_results/

## Assess calibrator generalisation
calibrator-generalisation: set-ceph-credentials set-gcp-credentials
	mkdir -p data/spectrum_data
	mkdir -p data/beam_preds
	mkdir -p models
	mkdir -p results
	# Copy the data from Ceph bucket to the local data directory
	aws s3 cp s3://winnow-g88rh/validation_datasets_corrected/spectrum_data/labelled/ data/spectrum_data/ --recursive --exclude "*" --include "dataset*" --profile winnow
	aws s3 cp s3://winnow-g88rh/validation_datasets_corrected/beam_preds/labelled/ data/beam_preds/ --recursive --exclude "*" --include "*-annotated_beam_preds.csv" --profile winnow
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
train-general-model:
	# Run the generalisation evaluation script
	winnow train --data-source instanovo --dataset-config-path configs/validation_data/train_general_model.yaml --model-output-folder general_model/model --dataset-output-path general_model/results/general_model_training_results.csv
	# Copy the results back to Ceph bucket
	aws s3 cp general_model/ s3://winnow-g88rh/general_model/ --recursive --profile winnow

## Evaluate general model on validation datasets
evaluate-general-model-validation-datasets: set-ceph-credentials set-gcp-credentials
	# Make folders
	mkdir -p validation_datasets_corrected/winnow_metadata/labelled
	mkdir -p validation_datasets_corrected/winnow_metadata/de_novo
	mkdir -p validation_datasets_corrected/beam_preds/labelled
	mkdir -p validation_datasets_corrected/beam_preds/de_novo
	mkdir -p validation_datasets_corrected/spectrum_data/labelled
	mkdir -p validation_datasets_corrected/spectrum_data/de_novo
	mkdir -p general_model/model
	# Copy the data from Ceph bucket to the local data directory
	aws s3 cp s3://winnow-g88rh/validation_datasets_corrected/beam_preds/labelled/ validation_datasets_corrected/beam_preds/labelled/ --recursive --profile winnow
	aws s3 cp s3://winnow-g88rh/validation_datasets_corrected/beam_preds/de_novo/ validation_datasets_corrected/beam_preds/de_novo/ --recursive --profile winnow
	aws s3 cp s3://winnow-g88rh/validation_datasets_corrected/spectrum_data/labelled/ validation_datasets_corrected/spectrum_data/labelled/ --recursive --profile winnow
	aws s3 cp s3://winnow-g88rh/validation_datasets_corrected/spectrum_data/de_novo/ validation_datasets_corrected/spectrum_data/de_novo/ --recursive --profile winnow
	aws s3 cp s3://winnow-g88rh/general_model/model/ general_model/model/ --recursive --profile winnow
	# Run the generalisation evaluation script
	# 1. Evaluate labelled data
	winnow predict --data-source instanovo --dataset-config-path configs/validation_data/test_general_model_validation_data.yaml --model-folder general_model/model --method winnow --fdr-threshold 1.0 --confidence-column "calibrated_confidence" --output-path validation_datasets_corrected/winnow_metadata/labelled/general_test_winnow_output.csv
	winnow predict --data-source instanovo --dataset-config-path configs/validation_data/test_general_model_validation_data.yaml --model-folder general_model/model --method database-ground --fdr-threshold 1.0 --confidence-column "calibrated_confidence" --output-path validation_datasets_corrected/winnow_metadata/labelled/general_test_dbg_output.csv
	# 2. Evaluate unlabelled data
	@for ds in $(val_datasets); do \
		winnow predict --data-source instanovo --dataset-config-path configs/validation_data/$${ds}_de_novo.yaml --model-folder general_model/model --method winnow --fdr-threshold 1.0 --confidence-column "calibrated_confidence" --output-path validation_datasets_corrected/winnow_metadata/de_novo/$${ds}_de_novo_preds.csv; \
	done
	# Copy the results back to Ceph bucket
	aws s3 cp validation_datasets_corrected/winnow_metadata/labelled/general_test_winnow_output.csv s3://winnow-g88rh/validation_datasets_corrected/winnow_metadata/labelled/ --profile winnow
	aws s3 cp validation_datasets_corrected/winnow_metadata/labelled/general_test_dbg_output.csv s3://winnow-g88rh/validation_datasets_corrected/winnow_metadata/labelled/ --profile winnow
	aws s3 cp validation_datasets_corrected/winnow_metadata/de_novo/ s3://winnow-g88rh/validation_datasets_corrected/winnow_metadata/de_novo/ --recursive --profile winnow
	# Copy to Google Cloud Storage
	gsutil -m cp -R validation_datasets_corrected/ gs://winnow-fdr/validation_datasets_corrected/

## Evaluate general model on external datasets
evaluate-general-model-external-datasets: set-ceph-credentials set-gcp-credentials
	# Make folders
	mkdir -p external_datasets/winnow_metadata/lcfm
	mkdir -p external_datasets/winnow_metadata/de_novo
	mkdir -p external_datasets/winnow_metadata/acfm
	mkdir -p external_datasets/beam_preds/lcfm
	mkdir -p external_datasets/beam_preds/de_novo
	mkdir -p external_datasets/beam_preds/acfm
	mkdir -p external_datasets/spectrum_data/lcfm
	mkdir -p external_datasets/spectrum_data/de_novo
	mkdir -p external_datasets/spectrum_data/acfm
	mkdir -p general_model/model
	# Copy the data from Ceph bucket to the local data directory
	aws s3 cp s3://winnow-g88rh/external_datasets/spectrum_data/lcfm/ external_datasets/spectrum_data/lcfm/ --recursive --profile winnow
	aws s3 cp s3://winnow-g88rh/external_datasets/beam_preds/lcfm/ external_datasets/beam_preds/lcfm/ --recursive --profile winnow
	aws s3 cp s3://winnow-g88rh/external_datasets/beam_preds/de_novo/ external_datasets/beam_preds/de_novo/ --recursive --profile winnow
	aws s3 cp s3://winnow-g88rh/external_datasets/spectrum_data/de_novo/ external_datasets/spectrum_data/de_novo/ --recursive --profile winnow
	aws s3 cp s3://winnow-g88rh/external_datasets/beam_preds/acfm/ external_datasets/beam_preds/acfm/ --recursive --profile winnow
	aws s3 cp s3://winnow-g88rh/external_datasets/spectrum_data/acfm/ external_datasets/spectrum_data/acfm/ --recursive --profile winnow
	aws s3 cp s3://winnow-g88rh/general_model/model/ general_model/model/ --recursive --profile winnow
	# Run the generalisation evaluation script
	# 1. Evaluate labelled data
	@for ds in $(ext_datasets); do \
	winnow predict --data-source instanovo --dataset-config-path configs/external_datasets/$${ds}_lcfm.yaml --model-folder general_model/model --method winnow --fdr-threshold 1.0 --confidence-column "calibrated_confidence" --output-path external_datasets/winnow_metadata/lcfm/$${ds}_lcfm_preds_winnow.csv; \
	winnow predict --data-source instanovo --dataset-config-path configs/external_datasets/$${ds}_lcfm.yaml --model-folder general_model/model --method database-ground --fdr-threshold 1.0 --confidence-column "calibrated_confidence" --output-path external_datasets/winnow_metadata/lcfm/$${ds}_lcfm_preds_dbg.csv; \
	done
	# 2. Evaluate de novo unlabelled data
	@for ds in $(ext_datasets); do \
		winnow predict --data-source instanovo --dataset-config-path configs/external_datasets/$${ds}_de_novo.yaml --model-folder general_model/model --method winnow --fdr-threshold 1.0 --confidence-column "calibrated_confidence" --output-path external_datasets/winnow_metadata/de_novo/$${ds}_de_novo_preds.csv; \
	done
	# 3. Evaluate raw unlabelled data
	@for ds in $(ext_datasets); do \
		winnow predict --data-source instanovo --dataset-config-path configs/external_datasets/$${ds}_acfm.yaml --model-folder general_model/model --method winnow --fdr-threshold 1.0 --confidence-column "calibrated_confidence" --output-path external_datasets/winnow_metadata/acfm/$${ds}_acfm_preds.csv; \
	done
	# Copy the results back to Ceph bucket
	aws s3 cp external_datasets/winnow_metadata/lcfm/ s3://winnow-g88rh/external_datasets/winnow_metadata/lcfm/ --recursive --profile winnow
	aws s3 cp external_datasets/winnow_metadata/de_novo/ s3://winnow-g88rh/external_datasets/winnow_metadata/de_novo/ --recursive --profile winnow
	aws s3 cp external_datasets/winnow_metadata/acfm/ s3://winnow-g88rh/external_datasets/winnow_metadata/acfm/ --recursive --profile winnow
	# Copy to Google Cloud Storage
	gsutil -m cp -R external_datasets/ gs://winnow-fdr/external_datasets/

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
add-db-fdr-external-datasets: set-ceph-credentials set-gcp-credentials
	# Make folders
	mkdir -p external_datasets/winnow_metadata/lcfm
	mkdir -p external_datasets/winnow_metadata/acfm
	# Copy the data from Ceph bucket to the local data directory
	aws s3 cp s3://winnow-g88rh/external_datasets/winnow_metadata/lcfm/ external_datasets/winnow_metadata/lcfm/ --recursive --profile winnow
	# Add database-grounded FDR to labelled data (LCFM)
	@for ds in $(ext_datasets); do \
		python scripts/add_db_fdr.py \
			--input-path external_datasets/winnow_metadata/lcfm/$${ds}_lcfm_preds_winnow.csv \
			--output-path external_datasets/winnow_metadata/lcfm/$${ds}_lcfm_preds.csv \
			--confidence-column calibrated_confidence; \
	done
	# Copy the results back to Ceph bucket
	aws s3 cp external_datasets/winnow_metadata/lcfm/ s3://winnow-g88rh/external_datasets/winnow_metadata/lcfm/ --recursive --profile winnow
	# Copy to Google Cloud Storage
	gsutil -m cp -R external_datasets/winnow_metadata/lcfm/ gs://winnow-fdr/external_datasets/winnow_metadata/lcfm/
