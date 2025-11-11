# This Makefile provides shortcut commands to facilitate local development.

# Common variables
PACKAGE_NAME = winnow

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
DOCKER_RUN = docker run $(DOCKER_RUN_FLAGS) $(DOCKER_IMAGE_NAME)

PYTEST = uv run pytest --verbose .

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

.PHONY: install install-frozen install-dev

## Install project dependencies (may resolve and update lockfile if needed)
# Note that uv includes the `dev` group by default, so we need to exclude it with `--no-dev`
install:
	uv sync --no-dev

## Install development tools (testing, linting, etc.)
install-dev:
	uv sync

## Install project dependencies exactly as specified in lockfile (fast, no resolution)
install-frozen:
	uv sync --frozen

## Install all dependencies
install-all:
	uv sync --all-extras

#################################################################################
## Development commands														 	#
#################################################################################

.PHONY: tests test-docker bash set-gcp-credentials set-ceph-credentials

## Run all tests
tests:
	$(PYTEST)

## Run all tests in the Docker Image
test-docker:
	docker run $(DOCKER_RUN_FLAGS) $(DOCKER_IMAGE) $(PYTEST)

## Open a bash shell in the Docker image
bash:
	docker run -it $(DOCKER_RUN_FLAGS) $(DOCKER_IMAGE) /bin/bash

## Set the GCP credentials
set-gcp-credentials:
	uv run python scripts/set_gcp_credentials.py
	gcloud auth activate-service-account dtu-denovo-sa@ext-dtu-denovo-sequencing-gcp.iam.gserviceaccount.com --key-file=ext-dtu-denovo-sequencing-gcp.json --project=ext-dtu-denovo-sequencing-gcp

## Set the Ceph credentials
set-ceph-credentials:
	uv run python scripts/set_ceph_credentials.py

# Download spectra from HuggingFace
download-spectra:
	uv run scripts/download_spectra.py

# Download beams from GCP
download-beams:
	gsutil -m cp \
		"gs://winnow-fdr/winnow-ms-datasets-new-outputs/celegans_labelled_beams.csv" \
		"gs://winnow-fdr/winnow-ms-datasets-new-outputs/celegans_raw_beams.csv" \
		"gs://winnow-fdr/winnow-ms-datasets-new-outputs/general_test_beams.csv" \
		"gs://winnow-fdr/winnow-ms-datasets-new-outputs/general_train_beams.csv" \
		"gs://winnow-fdr/winnow-ms-datasets-new-outputs/general_val_beams.csv" \
		"gs://winnow-fdr/winnow-ms-datasets-new-outputs/immuno2_labelled_beams.csv" \
		"gs://winnow-fdr/winnow-ms-datasets-new-outputs/immuno2_raw_beams.csv" \
		winnow-ms-datasets

# Train the general model
train-general-model:
	winnow train --data-source instanovo --dataset-config-path configs/train_general_model.yaml --model-output-dir general_model --dataset-output-path general_training_data.csv

# Upload the training results to GCP
upload-training-results:
	gsutil -m cp \
		general_training_data.csv \
		gs://winnow-fdr/winnow-ms-datasets-new-outputs/outputs/general_training_data.csv

# Upload the model to GCP
upload-model:
	gsutil -m cp \
		general_model/calibrator.pkl \
		gs://winnow-fdr/winnow-ms-datasets-new-outputs/general_model/calibrator.pkl

# Evaluate the general model
evaluate-general-model:
	winnow predict --data-source instanovo --dataset-config-path configs/evaluate_test_set.yaml --method winnow --fdr-threshold 1.0 --confidence-column calibrated_confidence --output-folder general_test_data/test_set --local-model-folder general_model
	winnow predict --data-source instanovo --dataset-config-path configs/evaluate_celegans_labelled.yaml --method winnow --fdr-threshold 1.0 --confidence-column calibrated_confidence --output-folder general_test_data/celegans_labelled --local-model-folder general_model
	winnow predict --data-source instanovo --dataset-config-path configs/evaluate_immuno2_labelled.yaml --method winnow --fdr-threshold 1.0 --confidence-column calibrated_confidence --output-folder general_test_data/immuno2_labelled --local-model-folder general_model
	winnow predict --data-source instanovo --dataset-config-path configs/evaluate_celegans_raw.yaml --method winnow --fdr-threshold 1.0 --confidence-column calibrated_confidence --output-folder general_test_data/celegans_raw --local-model-folder general_model
	winnow predict --data-source instanovo --dataset-config-path configs/evaluate_immuno2_raw.yaml --method winnow --fdr-threshold 1.0 --confidence-column calibrated_confidence --output-folder general_test_data/immuno2_raw --local-model-folder general_model

# Upload the evaluation results to GCP
upload-evaluation-results:
	gsutil -m cp \
		general_test_data/**/* \
		gs://winnow-fdr/winnow-ms-datasets-new-outputs/outputs/
