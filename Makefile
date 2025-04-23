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
DOCKERFILE_DEV := Dockerfile.dev
DOCKERFILE_CI := Dockerfile.ci

DOCKER_IMAGE_NAME = registry.gitlab.com/instadeep/dtu-denovo-sequencing
DOCKER_IMAGE_TAG = $(VERSION)
DOCKER_IMAGE = $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)
DOCKER_IMAGE_DEV = $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)-dev
DOCKER_IMAGE_CI = $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)
DOCKER_IMAGE_CI_DEV = $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)-dev

DOCKER_RUN_FLAGS = --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --shm-size='1gb'
DOCKER_RUN_FLAGS_VOLUME_MOUNT_HOME = $(DOCKER_RUN_FLAGS) --volume $(PWD):$(DOCKER_HOME_DIRECTORY)
DOCKER_RUN_FLAGS_VOLUME_MOUNT_RUNS = $(DOCKER_RUN_FLAGS) --volume $(PWD)/runs:$(DOCKER_RUNS_DIRECTORY)
DOCKER_RUN = docker run $(DOCKER_RUN_FLAGS) $(IMAGE_NAME)

PYTEST = pytest --alluredir=allure_results --cov-report=html --cov --cov-config=.coveragerc --random-order --verbose .
COVERAGE = coverage report -m

#################################################################################
## Docker build commands																#
#################################################################################

.PHONY: build build-arm build-dev build-dev-arm build-ci build-ci-dev

define docker_buildx_dev_template
	docker buildx build --platform=$(1) --progress=plain . \
		-f $(DOCKERFILE_DEV) -t $(2) --build-arg GID=$(shell id -g) \
		--build-arg UID=$(shell id -u) --build-arg LAST_COMMIT=$(LAST_COMMIT) \
		--build-arg VERSION=$(VERSION) --build-arg HOME_DIRECTORY=$(DOCKER_HOME_DIRECTORY) \
		--build-arg RUNS_DIRECTORY=$(DOCKER_RUNS_DIRECTORY)
endef

define docker_buildx_template
	docker buildx build --platform=$(1) --progress=plain . \
		-f $(DOCKERFILE) -t $(2) --build-arg GID=$(shell id -g) \
		--build-arg UID=$(shell id -u)  --build-arg LAST_COMMIT=$(LAST_COMMIT) \
		--build-arg VERSION=$(VERSION) --build-arg HOME_DIRECTORY=$(DOCKER_HOME_DIRECTORY) \
		--build-arg RUNS_DIRECTORY=$(DOCKER_RUNS_DIRECTORY)
endef

define docker_build_ci_template
	docker build --progress=plain . -f $(1) -t $(2) \
		--build-arg LAST_COMMIT=$(LAST_COMMIT) --build-arg VERSION=$(VERSION)
endef

## Build Docker image for InstaNovo on AMD64
build:
	$(call docker_buildx_template,linux/amd64,$(DOCKER_IMAGE))

## Build Docker image for InstaNovo on ARM64
build-arm:
	$(call docker_buildx_template,linux/arm64,$(DOCKER_IMAGE))

## Build development Docker image for InstaNovo on AMD64
build-dev:
	$(call docker_buildx_dev_template,linux/amd64,$(DOCKER_IMAGE_DEV))

## Build development Docker image for InstaNovo on ARM64
build-dev-arm:
	$(call docker_buildx_dev_template,linux/arm64,$(DOCKER_IMAGE_DEV))

## Build continuous integration Docker image for InstaNovo
build-ci:
	$(call docker_build_ci_template,$(DOCKERFILE),$(DOCKER_IMAGE_CI))
	docker tag $(DOCKER_IMAGE_CI) $(DOCKER_IMAGE)

## Build development continuous integration Docker image for InstaNovo
build-ci-dev:
	$(call docker_build_ci_template,$(DOCKERFILE_DEV),$(DOCKER_IMAGE_CI_DEV))
	docker tag $(DOCKER_IMAGE_CI_DEV) $(DOCKER_IMAGE_DEV)

#################################################################################
## Docker push commands																#
#################################################################################

.PHONY: push-ci push-ci-dev

## Push default and continuous integration Docker images for InstaNovo to GitLab registry
push-ci:
	docker push $(DOCKER_IMAGE)
	docker push $(DOCKER_IMAGE_CI)

## Push development and continuous integration development Docker images for InstaNovo to GitLab registry
push-ci-dev:
	docker push $(DOCKER_IMAGE_DEV)
	docker push $(DOCKER_IMAGE_CI_DEV)

#################################################################################
## Install packages commands																 	#
#################################################################################

.PHONY: compile install install-dev install-all

## Compile all the pinned requirements*.txt files from the unpinned requirements*.in files
compile:
	pip install --upgrade uv
	rm -f requirements/*.txt
	uv pip compile requirements/requirements.in --emit-index-url  --output-file=requirements/requirements.txt
	uv pip compile requirements/requirements-dev.in --output-file=requirements/requirements-dev.txt
	uv pip compile requirements/requirements-docs.in --output-file=requirements/requirements-docs.txt
	uv pip compile requirements/requirements-mlflow.in --output-file=requirements/requirements-mlflow.txt

## Install required packages
install:
	pip install --upgrade uv
	uv pip install -r requirements/requirements.txt

## Install required and development packages
install-dev:
	pip install --upgrade uv
	uv pip install -r requirements/requirements.txt \
	               -r requirements/requirements-dev.txt

## Install required, development, documentation and MLFlow packages
install-all:
	pip install --upgrade uv
	uv pip install -r requirements/requirements.txt \
	               -r requirements/requirements-dev.txt \
				   -r requirements/requirements-docs.txt \
				   -r requirements/requirements-mlflow.txt


##  Sync pinned dependencies with your virtual environment
sync:
	pip install --upgrade uv
	uv pip sync requirements/requirements.txt


#################################################################################
## Development commands														 	#
#################################################################################

.PHONY: tests coverage test-docker coverage-docker bash bash-dev docs set-gcp-credentials

## Run all tests
tests:
	python -m instanovo.scripts.get_zenodo_record
	$(PYTEST)

## Calculate the code coverage
coverage:
	$(COVERAGE)

## Run all tests in the development Docker Image
test-docker:
	docker run $(DOCKER_RUN_FLAGS) $(DOCKER_IMAGE_DEV) nvidia-smi && $(PYTEST)

## Calculate the code coverage in the development Docker image
coverage-docker:
	docker run $(DOCKER_RUN_FLAGS) $(DOCKER_IMAGE_DEV) nvidia-smi && $(PYTEST) && $(COVERAGE)

## Open a bash shell in the default Docker image
bash:
	docker run -it $(DOCKER_RUN_FLAGS) $(DOCKER_IMAGE) /bin/bash

## Open a bash shell in the development Docker image
bash-dev:
	docker run -it $(DOCKER_RUN_FLAGS) $(DOCKER_IMAGE_DEV) /bin/bash

## Serve the documentation site locally
docs:
	pip install --upgrade uv
	uv pip install -r requirements/requirements-docs.txt
	git config --global --add safe.directory "$(dirname "$(pwd)")"
	rm -rf docs/reference
	python ./docs/gen_ref_nav.py
	mkdocs build --verbose --site-dir docs_public
	mkdocs serve

## Set the GCP credentials
set-gcp-credentials:
	python scripts/set_gcp_credentials.py
	gcloud auth activate-service-account dtu-denovo-sa@ext-dtu-denovo-sequencing-gcp.iam.gserviceaccount.com --key-file=ext-dtu-denovo-sequencing-gcp.json --project=ext-dtu-denovo-sequencing-gcp

#################################################################################
## Validation dataset variables													#
#################################################################################

# List of all datasets to process
DATASETS := helaqc sbrodae herceptin immuno gluc snakevenoms woundfluids # tplantibodies

# Base directories
DATA_DIR := input_data
OUTPUT_DIR := output_data
MODEL_DIR := models

# Default parameters
TEST_FRACTION := 0.2
RANDOM_STATE := 42

# GCS paths for results
GCS_BASE := gs://winnow-fdr/validation_datasets_corrected
GCS_METADATA_LABELLED := $(GCS_BASE)/winnow_metadata/labelled
GCS_METADATA_DE_NOVO := $(GCS_BASE)/winnow_metadata/de_novo
GCS_CALIBRATOR_MODELS := $(GCS_BASE)/calibrator_models

# Config generation
CONFIG_DIR := configs
CONFIG_TEMPLATES := $(wildcard $(CONFIG_DIR)/*.yaml.template)

#################################################################################
## Condensed validation dataset commands										#
#################################################################################

# Variable to specify which dataset to process
DATASET ?= helaqc

# Validate that the specified dataset is in our list
validate-dataset:
	@if ! echo "$(DATASETS)" | grep -q "$(DATASET)"; then \
		echo "Error: Invalid dataset '$(DATASET)'. Must be one of: $(DATASETS)"; \
		exit 1; \
	fi

# Condensed preprocessing command
preprocess-dataset: validate-dataset
	@echo "Preprocessing dataset $(DATASET)"
	mkdir -p $(DATA_DIR)/beam_preds/labelled
	mkdir -p $(DATA_DIR)/beam_preds/de_novo
	mkdir -p $(DATA_DIR)/beam_preds/raw
	mkdir -p $(DATA_DIR)/spectrum_data/labelled
	mkdir -p $(DATA_DIR)/spectrum_data/de_novo
	mkdir -p $(DATA_DIR)/spectrum_data/raw
	gsutil cp $(GCS_BASE)/spectrum_data/labelled/dataset-$(DATASET)-annotated-0000-0001.parquet $(DATA_DIR)/spectrum_data/labelled/
	gsutil cp $(GCS_BASE)/beam_preds/raw/$(DATASET)_beam_preds.csv $(DATA_DIR)/beam_preds/raw/
	gsutil cp $(GCS_BASE)/spectrum_data/raw/dataset-$(DATASET)-raw-0000-0001.parquet $(DATA_DIR)/spectrum_data/raw/
	python scripts/preprocess_validation_data.py --species $(DATASET) --input_dir $(DATA_DIR)

# Condensed prepare command
prepare-dataset: preprocess-dataset
	@echo "Preparing dataset $(DATASET)"
	mkdir -p $(DATA_DIR)/splits/$(DATASET)
	python scripts/create_train_test_split.py \
		--spectrum_path $(DATA_DIR)/spectrum_data/labelled/dataset-$(DATASET)-annotated-0000-0001.parquet \
		--beam_predictions_path $(DATA_DIR)/beam_preds/labelled/$(DATASET)-annotated_beam_preds.csv \
		--output_dir $(DATA_DIR)/splits/$(DATASET) \
		--test_fraction $(TEST_FRACTION) \
		--random_state $(RANDOM_STATE)
	$(MAKE) generate-configs-dataset

# Condensed train and predict command
train-dataset: prepare-dataset
	@echo "Training and predicting on dataset $(DATASET)"
	mkdir -p $(MODEL_DIR)/$(DATASET)
	mkdir -p $(OUTPUT_DIR)/$(DATASET)

	# Train
	winnow train \
		--data-source=winnow \
		--dataset-config-path=$(CONFIG_DIR)/train-$(DATASET).yaml \
		--model-output-folder=$(MODEL_DIR)/$(DATASET) \
		--dataset-output-path=$(OUTPUT_DIR)/$(DATASET)/train_output.csv

	# Evaluate on labelled data using winnow method
	winnow predict \
		--data-source=winnow \
		--dataset-config-path=$(CONFIG_DIR)/predict_labelled-$(DATASET).yaml \
		--model-folder=$(MODEL_DIR)/$(DATASET) \
		--method=winnow \
		--fdr-threshold=0.05 \
		--confidence-column="calibrated_confidence" \
		--output-path=$(OUTPUT_DIR)/$(DATASET)/labelled_winnow_predict_output.csv

	# Evaluate on labelled data using database-ground method
	winnow predict \
		--data-source=winnow \
		--dataset-config-path=$(CONFIG_DIR)/predict_labelled-$(DATASET).yaml \
		--model-folder=$(MODEL_DIR)/$(DATASET) \
		--method=database-ground \
		--fdr-threshold=0.05 \
		--confidence-column="calibrated_confidence" \
		--output-path=$(OUTPUT_DIR)/$(DATASET)/labelled_database_ground_predict_output.csv

	# Evaluate on de novo data using winnow method
	winnow predict \
		--data-source=instanovo \
		--dataset-config-path=$(CONFIG_DIR)/predict_de_novo-$(DATASET).yaml \
		--model-folder=$(MODEL_DIR)/$(DATASET) \
		--method=winnow \
		--fdr-threshold=0.05 \
		--confidence-column="calibrated_confidence" \
		--output-path=$(OUTPUT_DIR)/$(DATASET)/de_novo_winnow_predict_output.csv

	# $(MAKE) copy-results-dataset

# Generic config generation command
generate-configs-dataset: validate-dataset
	@echo "Generating configs for dataset $(DATASET)"
	@bash -c 'for template in $(CONFIG_TEMPLATES); do \
		output=$$(basename $$template .yaml.template)-$(DATASET).yaml; \
		if [[ $$template == *"predict_de_novo"* ]]; then \
			sed "s/\$${DATASET}/$(DATASET)/g; s|beam_predictions_path:.*|beam_predictions_path: $(DATA_DIR)/beam_preds/de_novo/$(DATASET)_raw_beam_preds_filtered.csv|g; s|spectrum_path:.*|spectrum_path: $(DATA_DIR)/spectrum_data/de_novo/$(DATASET)_raw_filtered.parquet|g" $$template > $(CONFIG_DIR)/$$output; \
		else \
			sed "s/\$${DATASET}/$(DATASET)/g; s|data/|$(DATA_DIR)/|g" $$template > $(CONFIG_DIR)/$$output; \
		fi \
	done'

# Generic copy results command
copy-results-dataset: validate-dataset
	@echo "Copying results for $(DATASET) to GCP..."
	gsutil cp $(OUTPUT_DIR)/$(DATASET)/labelled_winnow_predict_output.csv $(GCS_METADATA_LABELLED)/$(DATASET)_test_labelled_winnow.csv
	gsutil cp $(OUTPUT_DIR)/$(DATASET)/labelled_database_ground_predict_output.csv $(GCS_METADATA_LABELLED)/$(DATASET)_test_labelled_database_ground.csv
	gsutil cp $(OUTPUT_DIR)/$(DATASET)/train_output.csv $(GCS_METADATA_LABELLED)/$(DATASET)_train_labelled.csv
	gsutil cp $(OUTPUT_DIR)/$(DATASET)/de_novo_winnow_predict_output.csv $(GCS_METADATA_DE_NOVO)/$(DATASET)_de_novo_preds.csv
	gsutil -m cp -r $(MODEL_DIR)/$(DATASET)/* $(GCS_CALIBRATOR_MODELS)/$(DATASET)/
	@echo "Results copied successfully!"

# Example usage:
# make preprocess-dataset DATASET=helaqc
# make prepare-dataset DATASET=helaqc
# make train-dataset DATASET=helaqc
# make generate-configs-dataset DATASET=helaqc
# make copy-results-dataset DATASET=helaqc

#################################################################################
## Validation dataset batch processing commands									#
#################################################################################

.PHONY: clean-all clean-configs process-all-datasets

# Process all datasets in sequence
process-all-datasets:
	@for dataset in $(DATASETS); do \
		echo "Processing dataset: $$dataset"; \
		$(MAKE) preprocess-dataset DATASET=$$dataset; \
		$(MAKE) prepare-dataset DATASET=$$dataset; \
		$(MAKE) train-dataset DATASET=$$dataset; \
	done

# Clean configs
clean-configs:
	@for dataset in $(DATASETS); do \
		rm -f $(CONFIG_DIR)/*-$$dataset.yaml; \
	done

# Clean all
clean-all: clean-configs
	rm -rf $(DATA_DIR) $(MODEL_DIR) $(OUTPUT_DIR)

# Example usage:
# make process-all-datasets
# make clean-all

#################################################################################
## External dataset variables													#
#################################################################################

# Base directories
DATA_DIR := input_data
OUTPUT_DIR := output_data
MODEL_DIR := models

# Default parameters
TEST_FRACTION := 0.2
RANDOM_STATE := 42

# GCS paths for results
GCS_BASE := gs://winnow-fdr/external_datasets
GCS_METADATA_LABELLED := $(GCS_BASE)/winnow_metadata/labelled
GCS_METADATA_DE_NOVO := $(GCS_BASE)/winnow_metadata/de_novo
GCS_CALIBRATOR_MODELS := $(GCS_BASE)/calibrator_models

# Config generation
CONFIG_DIR := configs
CONFIG_TEMPLATES := $(wildcard $(CONFIG_DIR)/*.yaml.template)

#################################################################################
## External validation dataset commands						    				#
#################################################################################

.PHONY: preprocess-external-datasets prepare-external-dataset generate-configs-external-dataset

# Condensed preprocessing command
preprocess-external-datasets:
	@echo "Preprocessing external datasets"
	mkdir -p $(DATA_DIR)/beam_preds/labelled
	mkdir -p $(DATA_DIR)/beam_preds/de_novo
	mkdir -p $(DATA_DIR)/beam_preds/raw
	mkdir -p $(DATA_DIR)/spectrum_data/labelled
	mkdir -p $(DATA_DIR)/spectrum_data/de_novo
	mkdir -p $(DATA_DIR)/spectrum_data/raw
	gsutil cp $(GCS_BASE)/spectrum_data/lcfm/*.parquet $(DATA_DIR)/spectrum_data/labelled/
	gsutil cp $(GCS_BASE)/beam_preds/acfm/*.csv $(DATA_DIR)/beam_preds/raw/
	gsutil cp $(GCS_BASE)/spectrum_data/acfm/*.parquet $(DATA_DIR)/spectrum_data/raw/
	python scripts/preprocess_external_data.py --input_dir $(DATA_DIR)

# Condensed prepare command
prepare-external-dataset: preprocess-external-dataset
	@echo "Preparing external datasets"
	mkdir -p $(DATA_DIR)/splits
	python scripts/create_train_test_split.py \
		--spectrum_path $(DATA_DIR)/spectrum_data/labelled/*.parquet \
		--beam_predictions_path $(DATA_DIR)/beam_preds/labelled/annotated_beam_preds.csv \
		--output_dir $(DATA_DIR)/splits \
		--test_fraction $(TEST_FRACTION) \
		--random_state $(RANDOM_STATE)
	$(MAKE) generate-configs-external-dataset

# Generic config generation command
generate-configs-external-dataset: validate-external-dataset
	@echo "Generating configs for external datasets"
	@bash -c 'for template in $(CONFIG_TEMPLATES); do \
		output=$$(basename $$template .yaml.template)-acfm.yaml; \
		sed "s/\$${DATASET}/acfm/g; s|data/|$(DATA_DIR)/|g" $$template > $(CONFIG_DIR)/$$output; \
	done'
