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
	features_path=null \
	dataset.spectrum_path_or_directory=examples/example_data/spectra.ipc \
	dataset.predictions_path=examples/example_data/predictions.csv \
	model_output_dir=models/new_model \
	irt_regressor_output_path=null \
	training_history_path=null \
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


#################################################################################
## Train general model commands													#
#################################################################################

.PHONY: train_general_model

train_general_model:
	uv run winnow train \
	features_path=train_shuffled/ \
	val_features_path=val_feature_matrices/PXD004424.parquet \
	calibrator.hidden_dims='[128,64]' \
	model_output_dir=general_model \
	dataset_output_path=general_model/calibrated_dataset.csv \
	training_history_path=general_model/training_history.json \
	calibrator.koina.input_columns.collision_energies=collision_energy \
	calibrator.koina.input_columns.fragmentation_types=frag_type \
	calibrator.learning_rate=0.0001 \
	calibrator.weight_decay=0.0001 \
	calibrator.max_epochs=200 \
	calibrator.batch_size=4096 \
	calibrator.n_iter_no_change=10 \
	calibrator.tol=1.0e-4 \
	calibrator.seed=42


#########################################################
## Evaluate general model commands
#########################################################

.PHONY: evaluate_general_model_annotated_biological_validation evaluate_general_model_raw_biological_validation evaluate_general_model_labelled_external_datasets evaluate_general_model_unlabelled_external_datasets annotate_preds_proteome_hits

BIOLOGICAL_VALIDATION_PROJECTS := gluc helaqc herceptin immuno sbrodae snakevenoms tplantibodies woundfluids

# Proteome FASTA paths for ``scripts/annotate_preds_proteome_hits.py`` (leave empty to skip).
# Outputs live under ``predictions/general_model/<project>_raw/`` or ``..._unlabelled/``.
PROTEOME_FASTA_raw_gluc := fasta/human.fasta
PROTEOME_FASTA_raw_helaqc := fasta/human.fasta
PROTEOME_FASTA_raw_herceptin := fasta/herceptin.fasta
PROTEOME_FASTA_raw_immuno := fasta/human.fasta
PROTEOME_FASTA_raw_sbrodae := fasta/Sb_proteome.fasta
PROTEOME_FASTA_raw_snakevenoms := fasta/uniprot-serpentes-2022.05.09.fasta
PROTEOME_FASTA_raw_tplantibodies := fasta/nanobody_library.fasta
PROTEOME_FASTA_raw_woundfluids := fasta/human.fasta

evaluate_general_model_annotated_biological_validation:
	for project in $(BIOLOGICAL_VALIDATION_PROJECTS); do \
		uv run winnow predict \
		dataset.spectrum_path_or_directory=held_out_projects/biological_validation/annotated/dataset-$$project-annotated-0000-0001.parquet \
		dataset.predictions_path=held_out_projects/biological_validation/annotated_predictions/dataset-$$project-annotated-0000-0001.csv \
		calibrator.pretrained_model_name_or_path=general_model \
		output_folder=predictions/general_model/$${project}_annotated/ \
		koina.input_constants.collision_energies=27 \
		koina.input_constants.fragmentation_types=HCD \
		fdr_control.fdr_threshold=1.0 \
		fdr_control.confidence_column=calibrated_confidence; \
	done

evaluate_general_model_raw_biological_validation:
	for project in $(BIOLOGICAL_VALIDATION_PROJECTS); do \
		uv run winnow predict \
		dataset.spectrum_path_or_directory=held_out_projects/biological_validation/raw/dataset-$$project-raw-0000-0001.parquet \
		dataset.predictions_path=held_out_projects/biological_validation/raw_predictions/dataset-$$project-raw-0000-0001.csv \
		calibrator.pretrained_model_name_or_path=general_model \
		output_folder=predictions/general_model/$${project}_raw/ \
		koina.input_constants.collision_energies=27 \
		koina.input_constants.fragmentation_types=HCD \
		fdr_control.fdr_threshold=1.0 \
		fdr_control.confidence_column=calibrated_confidence; \
	done
	uv run python scripts/annotate_preds_proteome_hits.py biological_validation_raw \
		$(foreach p,$(BIOLOGICAL_VALIDATION_PROJECTS),$(if $(strip $(PROTEOME_FASTA_raw_$(p))),--map $(p)=$(PROTEOME_FASTA_raw_$(p)),))

EXTERNAL_DATASETS := PXD009935 PXD014877 PXD023064

PROTEOME_FASTA_unlabelled_PXD009935 := fasta/human.fasta
PROTEOME_FASTA_unlabelled_PXD014877 := fasta/Celegans.fasta
PROTEOME_FASTA_unlabelled_PXD023064 := fasta/human.fasta

## Run proteome-hit annotation + short-peptide filtering only (uses PROTEOME_FASTA_* above).
annotate_preds_proteome_hits:
	uv run python scripts/annotate_preds_proteome_hits.py biological_validation_raw \
		$(foreach p,$(BIOLOGICAL_VALIDATION_PROJECTS),$(if $(strip $(PROTEOME_FASTA_raw_$(p))),--map $(p)=$(PROTEOME_FASTA_raw_$(p)),))
	uv run python scripts/annotate_preds_proteome_hits.py unlabelled_external \
		$(foreach p,$(EXTERNAL_DATASETS),$(if $(strip $(PROTEOME_FASTA_unlabelled_$(p))),--map $(p)=$(PROTEOME_FASTA_unlabelled_$(p)),))

evaluate_general_model_labelled_external_datasets:
	for project in $(EXTERNAL_DATASETS); do \
		uv run winnow predict \
		dataset.spectrum_path_or_directory=held_out_projects/lcfm/$$project/ \
		dataset.predictions_path=held_out_projects/lcfm/$${project}_predictions/$$project.csv \
		calibrator.pretrained_model_name_or_path=general_model \
		output_folder=predictions/general_model/$${project}_labelled/ \
		koina.input_columns.collision_energies=collision_energy \
		koina.input_columns.fragmentation_types=frag_type \
		fdr_control.fdr_threshold=1.0 \
		fdr_control.confidence_column=calibrated_confidence; \
	done

evaluate_general_model_unlabelled_external_datasets:
	for project in $(EXTERNAL_DATASETS); do \
		uv run winnow predict \
		dataset.spectrum_path_or_directory=held_out_projects/acfm/$$project/ \
		dataset.predictions_path=held_out_projects/acfm/$${project}_predictions/$$project.csv \
		calibrator.pretrained_model_name_or_path=general_model \
		output_folder=predictions/general_model/$${project}_unlabelled/ \
		koina.input_columns.collision_energies=collision_energy \
		koina.input_columns.fragmentation_types=frag_type \
		fdr_control.fdr_threshold=1.0 \
		fdr_control.confidence_column=calibrated_confidence; \
	done
	uv run python scripts/annotate_preds_proteome_hits.py unlabelled_external \
		$(foreach p,$(EXTERNAL_DATASETS),$(if $(strip $(PROTEOME_FASTA_unlabelled_$(p))),--map $(p)=$(PROTEOME_FASTA_unlabelled_$(p)),))
