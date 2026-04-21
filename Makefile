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

PYTEST = uv run pytest tests --verbose --cov=winnow --cov-report xml:coverage.xml --cov-report term-missing --junitxml=pytest.xml --cov-fail-under=0

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

.PHONY: tests clean-coverage test-docker bash build-package clean-build test-build check-build docs docs-serve clean-docs set-gcp-credentials

## Run all tests
tests:
	$(PYTEST)

## Clean coverage reports
clean-coverage:
	rm -rf htmlcov/ .coverage coverage.xml pytest.xml

## Run all tests in the Docker Image
test-docker:
	docker run $(DOCKER_RUN_FLAGS) $(DOCKER_IMAGE) $(PYTEST)

## Open a bash shell in the Docker image
bash:
	docker run -it $(DOCKER_RUN_FLAGS) $(DOCKER_IMAGE) /bin/bash

## Build the winnow-fdr package (creates wheel and sdist in dist/)
build-package:
	uv build

## Clean all build artifacts (dist/, build/, *.egg-info/)
clean-build:
	rm -rf dist/ build/ *.egg-info/ winnow_fdr.egg-info/

## Build the package cleanly from scratch (fails on errors, cleans up on success)
check-build: clean-build build-package
	@ls dist/*.whl dist/*.tar.gz >/dev/null 2>&1 && echo "Package build OK ✓" || (echo "Build produced no artifacts" && exit 1)
	$(MAKE) clean-build

## Build mkdocs site
docs:
	uv sync --group docs
	uv run mkdocs build --strict

## Serve mkdocs locally with live-reload
docs-serve:
	uv run mkdocs serve

## Remove mkdocs build output
clean-docs:
	rm -rf docs_public/

## Set the GCP credentials
set-gcp-credentials:
	uv run python scripts/set_gcp_credentials.py
	gcloud auth activate-service-account dtu-denovo-sa@ext-dtu-denovo-sequencing-gcp.iam.gserviceaccount.com --key-file=ext-dtu-denovo-sequencing-gcp.json --project=ext-dtu-denovo-sequencing-gcp

#################################################################################
## Sample data and CLI commands													#
#################################################################################

.PHONY: sample-data train-sample predict-sample clean clean-all

## Generate sample data files for testing
sample-data:
	uv run python scripts/generate_sample_data.py

## Run winnow train with sample data
train-sample:
	uv run winnow train \
	dataset.spectrum_path_or_directory=examples/example_data/spectra.ipc \
	dataset.predictions_path=examples/example_data/predictions.csv \
	model_output_dir=models/new_model \
	dataset_output_path=results/calibrated_dataset.csv \
	calibrator.features.retention_time_feature.train_fraction=0.3

## Run winnow predict with sample data (uses locally trained model from models/new_model)
predict-sample:
	uv run winnow predict \
	calibrator.pretrained_model_name_or_path=models/new_model \
	fdr_control.fdr_threshold=1.0 \
	dataset.spectrum_path_or_directory=examples/example_data/spectra.ipc \
	dataset.predictions_path=examples/example_data/predictions.csv

## Clean output directories (does not delete sample data)
clean:
	rm -rf models/ results/

## Clean outputs and regenerate sample data
clean-all: clean sample-data

#################################################################################
## General model commands														#
#################################################################################

.PHONY: copy_down_train_dataset compute_train_features

# PXD000561 PXD00900 PXD001258 done
# PROJECTS := PXD002052 PXD003868 PXD004325 \
#             PXD004424 PXD004452 PXD004467 PXD004536 PXD004732 PXD004947 \
#             PXD004948 PXD005025 PXD005573 PXD006201 PXD009449 \
#             PXD010154 PXD010595 PXD013868 PXD019483 PXD021013 PXD025859 \
#             PXD026629 PXD026649 PXD029360 PXD031025 PXD031032 PXD035158 \
#             PXD037009 PXD043989 PXD044301 PXD044325 PXD044641 PXD044830 \
#             PXD045299 PXD045457 PXD045471 PXD045662 PXD046182 PXD046460 \
#             PXD046802 PXD047134 PXD047641 PXD047761 PXD048219 PXD056559
# PROJECTS := PXD035158 PXD037009 PXD043989 PXD044301 PXD044325 PXD044641 PXD044830
PROJECTS := PXD003868 PXD004325 PXD004424 PXD004452 PXD004467 PXD004536 PXD004732 PXD004947 PXD009449 PXD021013 PXD056559

copy_down_train_dataset:
	aws s3 cp --recursive s3://winnow-g88rh/revisions/new_datasets/train/ data/train/
	aws s3 cp --recursive s3://winnow-g88rh/revisions/new_datasets/train_predictions/ data/train_predictions/

compute_train_features: copy_down_train_dataset
	@mkdir -p train_feature_matrices
	@for project in $(PROJECTS); do \
		echo "Computing features for $$project..."; \
		uv run winnow compute-features \
		dataset.spectrum_path_or_directory=data/train/$$project/ \
		dataset.predictions_path=data/train_predictions/$$project.csv \
		training_matrix_output_path=train_feature_matrices/$$project.parquet; \
		aws s3 cp train_feature_matrices/$$project.parquet s3://winnow-g88rh/revisions/new_datasets/train_feature_matrices/$$project.parquet; \
	done
