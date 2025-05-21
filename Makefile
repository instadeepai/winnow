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

.PHONY: tests coverage test-docker coverage-docker bash set-gcp-credentials set-ceph-credentials compare-classifiers

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

## Compare classifiers
compare-classifiers: set-ceph-credentials set-gcp-credentials
	# Copy the data from Ceph bucket to the local data directory
	mkdir -p data
	aws s3 cp s3://winnow-g88rh/validation_datasets_corrected/spectrum_data/labelled/train_spectrum_all_datasets.parquet data/ --profile winnow
	aws s3 cp s3://winnow-g88rh/validation_datasets_corrected/beam_preds/labelled/train_beam_all_datasets.csv data/ --profile winnow
	aws s3 cp s3://winnow-g88rh/validation_datasets_corrected/spectrum_data/labelled/val_spectrum_all_datasets.parquet data/ --profile winnow
	aws s3 cp s3://winnow-g88rh/validation_datasets_corrected/beam_preds/labelled/val_beam_all_datasets.csv data/ --profile winnow
	# Run the evaluation script
	python scripts/compare_classifiers.py --train-spectrum-path data/train_spectrum_all_datasets.parquet --train-predictions-path data/train_beam_all_datasets.csv --val-spectrum-path data/val_spectrum_all_datasets.parquet --val-predictions-path data/val_beam_all_datasets.csv --output-dir results/
	# Copy the results back to Ceph bucket
	# aws s3 cp results/ s3://winnow-g88rh/classifier_comparison/ --recursive --profile winnow
	# Copy the results back to Google Cloud Storage
	gsutil -m cp -R results/ gs://winnow-fdr/classifier_comparison/
