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
## Dataset variables															#
#################################################################################

# List of all datasets to process
DATASETS := helaqc sbrodae herceptin immuno

# Base directories
DATA_DIR := data
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

# Function to generate configs for a dataset
define generate-configs
$(foreach template,$(CONFIG_TEMPLATES),\
	$(foreach dataset,$(DATASETS),\
		$(CONFIG_DIR)/$(basename $(notdir $(template)))-$(dataset).yaml))
endef

# Rule to generate configs for a specific dataset
generate-configs-%:
	@echo "Generating configs for dataset $*"
	$(foreach template,$(CONFIG_TEMPLATES),\
		sed 's/\$${DATASET}/$*/g' $(template) > $(CONFIG_DIR)/$(basename $(notdir $(template)))-$*.yaml)

# Individual config generation rules
generate-configs-helaqc:
	@echo "Generating configs for dataset helaqc"
	@for template in $(CONFIG_TEMPLATES); do \
		output=$$(basename $$template .yaml.template)-helaqc.yaml; \
		sed "s/\$${DATASET}/helaqc/g" $$template > $(CONFIG_DIR)/$$output; \
	done

generate-configs-sbrodae:
	@echo "Generating configs for dataset sbrodae"
	@for template in $(CONFIG_TEMPLATES); do \
		output=$$(basename $$template .yaml.template)-sbrodae.yaml; \
		sed "s/\$${DATASET}/sbrodae/g" $$template > $(CONFIG_DIR)/$$output; \
	done

generate-configs-herceptin:
	@echo "Generating configs for dataset herceptin"
	@for template in $(CONFIG_TEMPLATES); do \
		output=$$(basename $$template .yaml.template)-herceptin.yaml; \
		sed "s/\$${DATASET}/herceptin/g" $$template > $(CONFIG_DIR)/$$output; \
	done

generate-configs-immuno:
	@echo "Generating configs for dataset immuno"
	@for template in $(CONFIG_TEMPLATES); do \
		output=$$(basename $$template .yaml.template)-immuno.yaml; \
		sed "s/\$${DATASET}/immuno/g" $$template > $(CONFIG_DIR)/$$output; \
	done

# Individual copy results rules
copy-results-helaqc:
	@echo "Copying results for helaqc to GCP..."
	gsutil cp $(DATA_DIR)/splits/helaqc/labelled_winnow_predict_output.csv $(GCS_METADATA_LABELLED)/helaqc_test_labelled.csv
	gsutil cp $(DATA_DIR)/splits/helaqc/train_output.csv $(GCS_METADATA_LABELLED)/helaqc_train_labelled.csv
	gsutil cp $(DATA_DIR)/splits/helaqc/de_novo_winnow_predict_output.csv $(GCS_METADATA_DE_NOVO)/helaqc_de_novo_preds.csv
	gsutil -m cp -r $(MODEL_DIR)/helaqc/* $(GCS_CALIBRATOR_MODELS)/helaqc/
	@echo "Results copied successfully!"

copy-results-sbrodae:
	@echo "Copying results for sbrodae to GCP..."
	gsutil cp $(DATA_DIR)/splits/sbrodae/labelled_winnow_predict_output.csv $(GCS_METADATA_LABELLED)/sbrodae_test_labelled.csv
	gsutil cp $(DATA_DIR)/splits/sbrodae/train_output.csv $(GCS_METADATA_LABELLED)/sbrodae_train_labelled.csv
	gsutil cp $(DATA_DIR)/splits/sbrodae/de_novo_winnow_predict_output.csv $(GCS_METADATA_DE_NOVO)/sbrodae_de_novo_preds.csv
	gsutil -m cp -r $(MODEL_DIR)/sbrodae/* $(GCS_CALIBRATOR_MODELS)/sbrodae/
	@echo "Results copied successfully!"

copy-results-herceptin:
	@echo "Copying results for herceptin to GCP..."
	gsutil cp $(DATA_DIR)/splits/herceptin/labelled_winnow_predict_output.csv $(GCS_METADATA_LABELLED)/herceptin_test_labelled.csv
	gsutil cp $(DATA_DIR)/splits/herceptin/train_output.csv $(GCS_METADATA_LABELLED)/herceptin_train_labelled.csv
	gsutil cp $(DATA_DIR)/splits/herceptin/de_novo_winnow_predict_output.csv $(GCS_METADATA_DE_NOVO)/herceptin_de_novo_preds.csv
	gsutil -m cp -r $(MODEL_DIR)/herceptin/* $(GCS_CALIBRATOR_MODELS)/herceptin/
	@echo "Results copied successfully!"

copy-results-immuno:
	@echo "Copying results for immuno to GCP..."
	gsutil cp $(DATA_DIR)/splits/immuno/labelled_winnow_predict_output.csv $(GCS_METADATA_LABELLED)/immuno_test_labelled.csv
	gsutil cp $(DATA_DIR)/splits/immuno/train_output.csv $(GCS_METADATA_LABELLED)/immuno_train_labelled.csv
	gsutil cp $(DATA_DIR)/splits/immuno/de_novo_winnow_predict_output.csv $(GCS_METADATA_DE_NOVO)/immuno_de_novo_preds.csv
	gsutil -m cp -r $(MODEL_DIR)/immuno/* $(GCS_CALIBRATOR_MODELS)/immuno/
	@echo "Results copied successfully!"

#################################################################################
## Dataset preparation and training targets										#
#################################################################################

.PHONY: prepare-all train-all clean-all prepare-helaqc prepare-sbrodae prepare-herceptin prepare-immuno train-helaqc train-sbrodae train-herceptin train-immuno copy-results-helaqc copy-results-sbrodae copy-results-herceptin copy-results-immuno generate-configs-helaqc generate-configs-sbrodae generate-configs-herceptin generate-configs-immuno

# Target to prepare all datasets
prepare-all: prepare-helaqc prepare-sbrodae prepare-herceptin prepare-immuno

# Target to train on all datasets
train-all: train-helaqc train-sbrodae train-herceptin train-immuno

# Individual prepare rules
prepare-helaqc:
	@echo "Preparing dataset helaqc"
	mkdir -p $(DATA_DIR)/splits/helaqc
	mkdir -p $(MODEL_DIR)/helaqc
	gsutil cp $(GCS_BASE)/beam_preds/labelled/helaqc-annotated_beam_preds.csv $(DATA_DIR)/
	gsutil cp $(GCS_BASE)/spectrum_data/labelled/dataset-helaqc-annotated-0000-0001.parquet $(DATA_DIR)/
	gsutil cp $(GCS_BASE)/beam_preds/de_novo/helaqc_raw_beam_preds_filtered.csv $(DATA_DIR)/
	gsutil cp $(GCS_BASE)/spectrum_data/de_novo/helaqc_raw_filtered.parquet $(DATA_DIR)/
	python scripts/create_train_test_split.py \
		--spectrum_path $(DATA_DIR)/dataset-helaqc-annotated-0000-0001.parquet \
		--beam_predictions_path $(DATA_DIR)/helaqc-annotated_beam_preds.csv \
		--output_dir $(DATA_DIR)/splits/helaqc \
		--test_fraction $(TEST_FRACTION) \
		--random_state $(RANDOM_STATE)
	$(MAKE) generate-configs-helaqc

prepare-sbrodae:
	@echo "Preparing dataset sbrodae"
	mkdir -p $(DATA_DIR)/splits/sbrodae
	mkdir -p $(MODEL_DIR)/sbrodae
	gsutil cp $(GCS_BASE)/beam_preds/labelled/sbrodae-annotated_beam_preds.csv $(DATA_DIR)/
	gsutil cp $(GCS_BASE)/spectrum_data/labelled/dataset-sbrodae-annotated-0000-0001.parquet $(DATA_DIR)/
	gsutil cp $(GCS_BASE)/beam_preds/de_novo/sbrodae_raw_beam_preds_filtered.csv $(DATA_DIR)/
	gsutil cp $(GCS_BASE)/spectrum_data/de_novo/sbrodae_raw_filtered.parquet $(DATA_DIR)/
	python scripts/create_train_test_split.py \
		--spectrum_path $(DATA_DIR)/dataset-sbrodae-annotated-0000-0001.parquet \
		--beam_predictions_path $(DATA_DIR)/sbrodae-annotated_beam_preds.csv \
		--output_dir $(DATA_DIR)/splits/sbrodae \
		--test_fraction $(TEST_FRACTION) \
		--random_state $(RANDOM_STATE)
	$(MAKE) generate-configs-sbrodae

prepare-herceptin:
	@echo "Preparing dataset herceptin"
	mkdir -p $(DATA_DIR)/splits/herceptin
	mkdir -p $(MODEL_DIR)/herceptin
	gsutil cp $(GCS_BASE)/beam_preds/labelled/herceptin-annotated_beam_preds.csv $(DATA_DIR)/
	gsutil cp $(GCS_BASE)/spectrum_data/labelled/dataset-herceptin-annotated-0000-0001.parquet$(DATA_DIR)/
	gsutil cp $(GCS_BASE)/beam_preds/de_novo/herceptin_raw_beam_preds_filtered.csv $(DATA_DIR)/
	gsutil cp $(GCS_BASE)/spectrum_data/de_novo/herceptin_raw_filtered.parquet $(DATA_DIR)/
	python scripts/create_train_test_split.py \
		--spectrum_path $(DATA_DIR)/dataset-herceptin-annotated-0000-0001.parquet \
		--beam_predictions_path $(DATA_DIR)/herceptin-annotated_beam_preds.csv \
		--output_dir $(DATA_DIR)/splits/herceptin \
		--test_fraction $(TEST_FRACTION) \
		--random_state $(RANDOM_STATE)
	$(MAKE) generate-configs-herceptin

prepare-immuno:
	@echo "Preparing dataset immuno"
	mkdir -p $(DATA_DIR)/splits/immuno
	mkdir -p $(MODEL_DIR)/immuno
	gsutil cp $(GCS_BASE)/beam_preds/labelled/immuno-annotated_beam_preds.csv $(DATA_DIR)/
	gsutil cp $(GCS_BASE)/spectrum_data/labelled/dataset-immuno-annotated-0000-0001.parquet $(DATA_DIR)/
	gsutil cp $(GCS_BASE)/beam_preds/de_novo/immuno_raw_beam_preds_filtered.csv $(DATA_DIR)/
	gsutil cp $(GCS_BASE)/spectrum_data/de_novo/immuno_raw_filtered.parquet $(DATA_DIR)/
	python scripts/create_train_test_split.py \
		--spectrum_path $(DATA_DIR)/dataset-immuno-annotated-0000-0001.parquet \
		--beam_predictions_path $(DATA_DIR)/immuno-annotated_beam_preds.csv \
		--output_dir $(DATA_DIR)/splits/immuno \
		--test_fraction $(TEST_FRACTION) \
		--random_state $(RANDOM_STATE)
	$(MAKE) generate-configs-immuno

# Individual train rules
train-helaqc: prepare-helaqc
	@echo "Training on dataset helaqc"
	mkdir -p $(MODEL_DIR)/helaqc
	chmod +x run.sh
	./run.sh $(MODEL_DIR)/helaqc $(DATA_DIR)/splits/helaqc $(CONFIG_DIR)/train-helaqc.yaml $(CONFIG_DIR)/predict_labelled-helaqc.yaml $(CONFIG_DIR)/predict_de_novo-helaqc.yaml
	$(MAKE) copy-results-helaqc

train-sbrodae: prepare-sbrodae
	@echo "Training on dataset sbrodae"
	mkdir -p $(MODEL_DIR)/sbrodae
	chmod +x run.sh
	./run.sh $(MODEL_DIR)/sbrodae $(DATA_DIR)/splits/sbrodae $(CONFIG_DIR)/train-sbrodae.yaml $(CONFIG_DIR)/predict_labelled-sbrodae.yaml $(CONFIG_DIR)/predict_de_novo-sbrodae.yaml
	$(MAKE) copy-results-sbrodae

train-herceptin: prepare-herceptin
	@echo "Training on dataset herceptin"
	mkdir -p $(MODEL_DIR)/herceptin
	chmod +x run.sh
	./run.sh $(MODEL_DIR)/herceptin $(DATA_DIR)/splits/herceptin $(CONFIG_DIR)/train-herceptin.yaml $(CONFIG_DIR)/predict_labelled-herceptin.yaml $(CONFIG_DIR)/predict_de_novo-herceptin.yaml
	$(MAKE) copy-results-herceptin

train-immuno: prepare-immuno
	@echo "Training on dataset immuno"
	mkdir -p $(MODEL_DIR)/immuno
	chmod +x run.sh
	./run.sh $(MODEL_DIR)/immuno $(DATA_DIR)/splits/immuno $(CONFIG_DIR)/train-immuno.yaml $(CONFIG_DIR)/predict_labelled-immuno.yaml $(CONFIG_DIR)/predict_de_novo-immuno.yaml
	$(MAKE) copy-results-immuno

# Clean configs
clean-configs:
	rm -f $(CONFIG_DIR)/*-$(DATASETS).yaml

# Update clean-all to include configs
clean-all: clean-configs
	rm -rf $(DATA_DIR) $(MODEL_DIR)

# Individual dataset targets
evaluate-helaqc: train-helaqc
evaluate-sbrodae: train-sbrodae
evaluate-herceptin: train-herceptin
evaluate-immuno: train-immuno
