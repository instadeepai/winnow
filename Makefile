# This Makefile provides shortcut commands to facilitate local development.

# Common variables
PACKAGE_NAME = winnow

# Train variables
NUM_NODES = 1
BATCH_SIZE = 12
NUM_GPUS:= $(shell python -m instanovo.scripts.parse_nr_gpus)


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
## Dataset variables																#
#################################################################################

# List of all datasets to process
DATASETS := helaqc sbrodae herceptin immuno

# Base directories
DATA_DIR := data
MODEL_DIR := runs

# Default parameters
TEST_FRACTION := 0.2
RANDOM_STATE := 42

# GCS paths
GCS_BASE := gs://winnow-fdr/validation_datasets_corrected
GCS_BEAM_PREDS_LABELLED := $(GCS_BASE)/beam_preds/labelled/%-annotated_beam_preds.csv
GCS_SPECTRUM_LABELLED := $(GCS_BASE)/spectrum_data/labelled/dataset-%-annotated-0000-0001.parquet
GCS_BEAM_PREDS_DE_NOVO := $(GCS_BASE)/beam_preds/de_novo/%-raw_beam_preds_filtered.csv
GCS_SPECTRUM_DE_NOVO := $(GCS_BASE)/spectrum_data/de_novo/%-raw_filtered.parquet

# GCS paths for results
GCS_METADATA_LABELLED := gs://winnow-fdr/validation_datasets_corrected/winnow_metadata/labelled
GCS_METADATA_DE_NOVO := gs://winnow-fdr/validation_datasets_corrected/winnow_metadata/de_novo
GCS_CALIBRATOR_MODELS := gs://winnow-fdr/validation_datasets_corrected/calibrator_models

# Config generation
CONFIG_DIR := configs
CONFIG_TEMPLATES := $(wildcard $(CONFIG_DIR)/*.yaml.template)

# Pattern rule to generate configs for each dataset
$(CONFIG_DIR)/%-$(DATASET).yaml: $(CONFIG_DIR)/%.yaml.template
	sed 's/\$${DATASET}/$(DATASET)/g' $< > $@

# Function to generate all configs for a dataset
define generate-configs
$(foreach template,$(CONFIG_TEMPLATES),\
	$(eval CONFIG_NAME=$(basename $(notdir $(template)))-$(1).yaml)\
	$(CONFIG_DIR)/$(CONFIG_NAME))
endef

#################################################################################
## Dataset preparation and training targets										#
#################################################################################

.PHONY: prepare-all train-all clean-all $(addprefix prepare-,$(DATASETS)) $(addprefix train-,$(DATASETS))

# Target to prepare all datasets
prepare-all: $(addprefix prepare-,$(DATASETS))

# Target to train on all datasets
train-all: $(addprefix train-,$(DATASETS))

# Pattern rule for preparing each dataset
prepare-%:
	# Create necessary directories
	mkdir -p $(DATA_DIR)/splits/$*
	mkdir -p $(MODEL_DIR)/$*
	# Download labelled data
	gsutil cp $(GCS_BEAM_PREDS_LABELLED) $(DATA_DIR)/
	gsutil cp $(GCS_SPECTRUM_LABELLED) $(DATA_DIR)/
	# Download de novo data
	gsutil cp $(GCS_BEAM_PREDS_DE_NOVO) $(DATA_DIR)/
	gsutil cp $(GCS_SPECTRUM_DE_NOVO) $(DATA_DIR)/
	# Create train/test split
	python scripts/create_train_test_split.py \
		--spectrum_path $(DATA_DIR)/dataset-$*-annotated-0000-0001.parquet \
		--beam_predictions_path $(DATA_DIR)/$*-annotated_beam_preds.csv \
		--output_dir $(DATA_DIR)/splits/$* \
		--test_fraction $(TEST_FRACTION) \
		--random_state $(RANDOM_STATE)
	# Generate configs for this dataset
	$(MAKE) $(call generate-configs,$*)

# Pattern rule for copying results to GCP
copy-results-%:
	@echo "Copying results for $* to GCP..."
	# Copy labelled data results
	gsutil cp $(DATA_DIR)/splits/$*/labelled_winnow_predict_output.csv $(GCS_METADATA_LABELLED)/$*_test_labelled.csv
	gsutil cp $(DATA_DIR)/splits/$*/train_output.csv $(GCS_METADATA_LABELLED)/$*_train_labelled.csv
	# Copy de novo results
	gsutil cp $(DATA_DIR)/splits/$*/de_novo_winnow_predict_output.csv $(GCS_METADATA_DE_NOVO)/$*_de_novo_preds.csv
	# Copy model to GCP
	gsutil -m cp -r $(MODEL_DIR)/$*/* $(GCS_CALIBRATOR_MODELS)/$*/
	@echo "Results copied successfully!"

# Pattern rule for training on each dataset
train-%: prepare-%
	mkdir -p $(MODEL_DIR)/$*
	chmod +x run.sh
	./run.sh $(MODEL_DIR)/$* $(DATA_DIR)/splits/$* $(CONFIG_DIR)/train-$*.yaml $(CONFIG_DIR)/predict_labelled-$*.yaml $(CONFIG_DIR)/predict_de_novo-$*.yaml
	$(MAKE) copy-results-$*

# Clean configs
clean-configs:
	rm -f $(CONFIG_DIR)/*-$(DATASET).yaml

# Update clean-all to include configs
clean-all: clean-configs
	rm -rf $(DATA_DIR) $(MODEL_DIR)

# Individual dataset targets
evaluate-helaqc: train-helaqc
evaluate-sbrodae: train-sbrodae
evaluate-herceptin: train-herceptin
evaluate-immuno: train-immuno
