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
	gsutil -m cp -r \
		general_test_data/* \
		gs://winnow-fdr/winnow-ms-datasets-new-outputs/outputs/

download-transformer-only-general-model:
	gsutil -m cp gs://winnow-fdr/winnow-ms-datasets-new-outputs/transformer_only_general_model/calibrator.pkl transformer_only_general_model/calibrator.pkl

download-general-model:
	gsutil -m cp gs://winnow-fdr/winnow-ms-datasets-new-outputs/general_model/calibrator.pkl general_model/calibrator.pkl

download-transformer-only-evaluation-outputs:
	mkdir transformer_only_outputs
	gsutil -m cp -r \
		"gs://winnow-fdr/winnow-ms-datasets-new-outputs/transformer_only_outputs/celegans_labelled" \
		"gs://winnow-fdr/winnow-ms-datasets-new-outputs/transformer_only_outputs/celegans_raw" \
		"gs://winnow-fdr/winnow-ms-datasets-new-outputs/transformer_only_outputs/general_training_data.csv" \
		"gs://winnow-fdr/winnow-ms-datasets-new-outputs/transformer_only_outputs/immuno2_labelled" \
		"gs://winnow-fdr/winnow-ms-datasets-new-outputs/transformer_only_outputs/immuno2_raw" \
		"gs://winnow-fdr/winnow-ms-datasets-new-outputs/transformer_only_outputs/test_set" \
		transformer_only_outputs

download-evaluation-outputs:
	mkdir outputs
	gsutil -m cp -r \
		"gs://winnow-fdr/winnow-ms-datasets-new-outputs/outputs/celegans_labelled" \
		"gs://winnow-fdr/winnow-ms-datasets-new-outputs/outputs/celegans_raw" \
		"gs://winnow-fdr/winnow-ms-datasets-new-outputs/outputs/general_training_data.csv" \
		"gs://winnow-fdr/winnow-ms-datasets-new-outputs/outputs/immuno2_labelled" \
		"gs://winnow-fdr/winnow-ms-datasets-new-outputs/outputs/immuno2_raw" \
		"gs://winnow-fdr/winnow-ms-datasets-new-outputs/outputs/test_set" \
		outputs

#################################################################################
## Merco commands														 	#
#################################################################################

MARCO_FOLDERS = PA NB8 NB6 NB5 NB4 NB3 NB2 NB13 NB12 NB10 NB1 MA3 MA2 MA1 BSA BIND17 BIND16 BIND15

# Generate phony targets for all individual folder commands
MARCO_DOWNLOAD_TARGETS = $(addprefix download_marco_dataset_,$(MARCO_FOLDERS))
MARCO_EVALUATE_TARGETS = $(addprefix evaluate_marco_dataset_,$(MARCO_FOLDERS))
MARCO_TRANSFER_TARGETS = $(addprefix transfer_marco_dataset_,$(MARCO_FOLDERS))
MARCO_EVALUATE_TRANSFORMER_ONLY_TARGETS = $(addprefix evaluate_marco_transformer_only_dataset_,$(MARCO_FOLDERS))
MARCO_TRANSFER_TRANSFORMER_ONLY_TARGETS = $(addprefix transfer_marco_transformer_only_dataset_,$(MARCO_FOLDERS))

.PHONY: download_marco_dataset evaluate_marco_dataset transfer_marco_dataset evaluate_marco_transformer_only_dataset transfer_marco_transformer_only_dataset set-foundation-model-ceph-credentials download-model
.PHONY: $(MARCO_DOWNLOAD_TARGETS) $(MARCO_EVALUATE_TARGETS) $(MARCO_TRANSFER_TARGETS) $(MARCO_EVALUATE_TRANSFORMER_ONLY_TARGETS) $(MARCO_TRANSFER_TRANSFORMER_ONLY_TARGETS)

## Set the Foundation Model Ceph credentials
set-foundation-model-ceph-credentials:
	uv run python scripts/set_foundation_model_ceph_credentials.py

## Download all Marco dataset folders
download_marco_dataset:
	aws s3 cp s3://mass-spec-foundation-model-8f55w/to_run/marco_instanovo_outputs/ marco_instanovo_outputs --recursive --profile foundation_model --recursive --profile foundation_model

## Download all Marco InstaNovo outputs
download_marco_beams:
	aws s3 cp s3://mass-spec-foundation-model-8f55w/to_run/marco_instanovo_outputs/ marco_beams --recursive --profile foundation_model

## Download the general (diffusion) model
download-model:
	gsutil -m cp gs://winnow-fdr/winnow-ms-datasets-new-outputs/general_model/calibrator.pkl general_model/calibrator.pkl

# Download the transformer only model
download-model:
	gsutil -m cp gs://winnow-fdr/winnow-ms-datasets-new-outputs/transformer_only_general_model/calibrator.pkl transformer_only_general_model/calibrator.pkl

## Evaluate all Marco dataset folders
evaluate_marco_dataset:
	$(foreach folder,$(MARCO_FOLDERS),$(MAKE) evaluate_marco_dataset_$(folder);)

## Evaluate all Marco dataset folders using the transformer only model
evaluate_marco_transformer_only_dataset:
	$(foreach folder,$(MARCO_FOLDERS),$(MAKE) evaluate_marco_transformer_only_dataset_$(folder);)

## Transfer all Marco Winnow outputs
transfer_marco_dataset:
	$(foreach folder,$(MARCO_FOLDERS),$(MAKE) transfer_marco_dataset_$(folder);)

## Transfer all Marco Transformer Only Winnow outputs
transfer_marco_transformer_only_dataset:
	$(foreach folder,$(MARCO_FOLDERS),$(MAKE) transfer_marco_transformer_only_dataset_$(folder);)

# Generate explicit rules for each folder
define MARCO_DOWNLOAD_RULE
download_marco_dataset_$(1):
	@mkdir -p marco
	aws s3 cp s3://mass-spec-foundation-model-8f55w/to_run/marco/$(1)/ marco/$(1)/ --recursive --profile foundation_model
	aws s3 cp s3://mass-spec-foundation-model-8f55w/to_run/marco_instanovo_outputs/$(1)_beams.csv marco_beams/$(1)_beams.csv --profile foundation_model
endef

# Evalaute using the general diffusion model
define MARCO_EVALUATE_RULE
evaluate_marco_dataset_$(1):
	winnow predict --data-source configs/marco/$(1).yaml --method winnow --fdr-threshold 1.0 --confidence-column calibrated_confidence --output-folder marco_winnow_outputs/$(1) --local-model-folder general_model
endef

# Evaluate using the transformer only model
define MARCO_EVALUATE_TRANSFORMER_ONLY_RULE
evaluate_marco_transformer_only_dataset_$(1):
	winnow predict --data-source configs/marco/$(1).yaml --method winnow --fdr-threshold 1.0 --confidence-column calibrated_confidence --output-folder marco_transformer_only_winnow_outputs/$(1) --local-model-folder transformer_only_general_model
endef

# Transfer winnow outputs to the foundation model bucket
define MARCO_TRANSFER_RULE
transfer_marco_dataset_$(1):
	aws s3 cp marco_winnow_outputs/$(1)/ s3://mass-spec-foundation-model-8f55w/to_run/marco_winnow_outputs/$(1)/ --recursive --profile foundation_model
endef

# Transfer transformer only outputs to the foundation model bucket
define MARCO_TRANSFER_TRANSFORMER_ONLY_RULE
transfer_marco_transformer_only_dataset_$(1):
	aws s3 cp transformer_only_outputs/$(1)/ s3://mass-spec-foundation-model-8f55w/to_run/marco_transformer_only_winnow_outputs/$(1)/ --recursive --profile foundation_model
endef

$(foreach folder,$(MARCO_FOLDERS),$(eval $(call MARCO_DOWNLOAD_RULE,$(folder))))
$(foreach folder,$(MARCO_FOLDERS),$(eval $(call MARCO_EVALUATE_RULE,$(folder))))
$(foreach folder,$(MARCO_FOLDERS),$(eval $(call MARCO_EVALUATE_TRANSFORMER_ONLY_RULE,$(folder))))
$(foreach folder,$(MARCO_FOLDERS),$(eval $(call MARCO_TRANSFER_RULE,$(folder))))
$(foreach folder,$(MARCO_FOLDERS),$(eval $(call MARCO_TRANSFER_TRANSFORMER_ONLY_RULE,$(folder))))
