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
DOCKERFILE_KOINA := Dockerfile.koina

DOCKER_IMAGE_NAME = winnow
DOCKER_IMAGE_TAG = $(VERSION)
DOCKER_IMAGE = $(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)

DOCKER_KOINA_IMAGE_NAME = winnow-koina
DOCKER_KOINA_IMAGE = $(DOCKER_KOINA_IMAGE_NAME):$(DOCKER_IMAGE_TAG)

DOCKER_RUN_FLAGS = --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --shm-size='1gb'
DOCKER_RUN_FLAGS_VOLUME_MOUNT_HOME = $(DOCKER_RUN_FLAGS) --volume $(PWD):$(DOCKER_HOME_DIRECTORY)
DOCKER_RUN_FLAGS_VOLUME_MOUNT_RUNS = $(DOCKER_RUN_FLAGS) --volume $(PWD)/runs:$(DOCKER_RUNS_DIRECTORY)
DOCKER_RUN = docker run $(DOCKER_RUN_FLAGS) $(DOCKER_IMAGE_NAME)

# Koina/Triton image needs a larger /dev/shm and exposes Triton ports for local poking.
DOCKER_RUN_FLAGS_KOINA = --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --shm-size='8gb' -p 8500-8502:8500-8502

PYTEST = uv run pytest tests --verbose --cov=winnow --cov-report xml:coverage.xml --cov-report term-missing --junitxml=pytest.xml --cov-fail-under=0

# --- Koina inference server overrides ---
# Defaults target the public Koina server so that local invocations work out of
# the box. When running inside the winnow-koina image (Triton is bundled and
# started by entrypoint.koina.sh), override to the in-pod gRPC endpoint, e.g.
#   make compute_train_features KOINA_SERVER_URL=localhost:8500 KOINA_SSL=false
# (manifest.yaml sets these overrides via spec.command for AIChor experiments.)
KOINA_SERVER_URL ?= koina.wilhelmlab.org:443
KOINA_SSL ?= true

KOINA_OVERRIDES = koina.server_url=$(KOINA_SERVER_URL) \
                  koina.ssl=$(KOINA_SSL)

# predict.yaml has local defaults (model path, irt_regressor_path, etc.).
# Makefile eval targets override everything explicitly so they work
# regardless of local config.

# Null out predict.yaml defaults that don't apply to eval targets.
PREDICT_EVAL_OVERRIDES = calibrator.irt_regressor_path=null

# Biological validation datasets have no per-row CE/frag metadata:
KOINA_FRAGMENT_MATCH_CONSTANTS = +koina.input_constants.collision_energies=27 \
                          +koina.input_constants.fragmentation_types=HCD \
						  +calibrator.features.fragment_match_features.model_input_constants.collision_energies=27 \
						  +calibrator.features.fragment_match_features.model_input_constants.fragmentation_types=HCD

# External / Astral datasets carry per-row metadata columns:
KOINA_FRAGMENT_MATCH_COLUMNS = +koina.input_columns.collision_energies=collision_energy \
                        +koina.input_columns.fragmentation_types=frag_type \
						+calibrator.features.fragment_match_features.model_input_columns.collision_energies=collision_energy \
						+calibrator.features.fragment_match_features.model_input_columns.fragmentation_types=frag_type

#################################################################################
## Docker build commands													    #
#################################################################################

.PHONY: build build-arm build-koina bash-koina

define docker_buildx_template
	docker buildx build --platform=$(1) --progress=plain . \
		-f $(2) -t $(3) --build-arg GID=$(shell id -g) \
		--build-arg UID=$(shell id -u)  --build-arg LAST_COMMIT=$(LAST_COMMIT) \
		--build-arg VERSION=$(VERSION) --build-arg HOME_DIRECTORY=$(DOCKER_HOME_DIRECTORY) \
		--build-arg RUNS_DIRECTORY=$(DOCKER_RUNS_DIRECTORY)
endef

## Build Docker image for winnow on AMD64
build:
	$(call docker_buildx_template,linux/amd64,$(DOCKERFILE),$(DOCKER_IMAGE))

## Build Docker image for winnow on ARM64
build-arm:
	$(call docker_buildx_template,linux/arm64,$(DOCKERFILE),$(DOCKER_IMAGE))

## Build the Koina-bundled Docker image (Triton + winnow) on AMD64
build-koina:
	$(call docker_buildx_template,linux/amd64,$(DOCKERFILE_KOINA),$(DOCKER_KOINA_IMAGE))

## Open a bash shell in the Koina-bundled Docker image (Triton starts in the background)
bash-koina:
	docker run -it $(DOCKER_RUN_FLAGS_KOINA) $(DOCKER_KOINA_IMAGE) /bin/bash

#################################################################################
## In-pod Triton/Koina control (interactive use only)							#
#################################################################################
# When AIChor's vscode debug is enabled the container's ENTRYPOINT (which would
# normally start Triton in the background) is replaced by the vscode tunnel, so
# the in-pod Koina server is never brought up. Run `make koina-up` once at the
# start of a debug session before any winnow target that talks to Koina.

.PHONY: koina-up koina-down koina-status

KOINA_PID_FILE := /tmp/koina.pid
KOINA_LOG_FILE := /tmp/koina.log
KOINA_HEALTH_URL := http://localhost:8501/v2/health/ready
KOINA_READY_TIMEOUT_SECS ?= 1800

## Start Triton/Koina in the background (no-op if already healthy)
koina-up:
	@if curl -fsS $(KOINA_HEALTH_URL) >/dev/null 2>&1; then \
		echo "[koina-up] Koina already healthy at $(KOINA_HEALTH_URL)"; \
		exit 0; \
	fi
	@if [ ! -x /models/start.py ]; then \
		echo "[koina-up] /models/start.py not found - is this the winnow-koina image?"; \
		exit 1; \
	fi
	@echo "[koina-up] starting Triton/Koina (logs: $(KOINA_LOG_FILE))..."
	@# Triton's Python backend stub doesn't pick up /usr/local/lib/python3.10/dist-packages
	@# from its computed sys.path; force it via PYTHONPATH so numpy / pandas / etc. import.
	@PYTHONPATH=/usr/local/lib/python3.10/dist-packages$${PYTHONPATH:+:$$PYTHONPATH} \
		nohup /models/start.py >$(KOINA_LOG_FILE) 2>&1 & echo $$! > $(KOINA_PID_FILE)
	@echo "[koina-up] waiting for $(KOINA_HEALTH_URL) (timeout=$(KOINA_READY_TIMEOUT_SECS)s)..."
	@DEADLINE=$$(( $$(date +%s) + $(KOINA_READY_TIMEOUT_SECS) )); \
	until curl -fsS $(KOINA_HEALTH_URL) >/dev/null 2>&1; do \
		KPID=$$(cat $(KOINA_PID_FILE)); \
		if ! kill -0 $$KPID 2>/dev/null; then \
			echo "[koina-up] Triton exited before becoming ready (last 200 lines of log):"; \
			tail -n 200 $(KOINA_LOG_FILE) >&2 || true; \
			exit 1; \
		fi; \
		if [ $$(date +%s) -gt $$DEADLINE ]; then \
			echo "[koina-up] timeout after $(KOINA_READY_TIMEOUT_SECS)s waiting for Koina readiness" >&2; \
			exit 1; \
		fi; \
		sleep 5; \
	done
	@echo "[koina-up] Koina is ready (pid=$$(cat $(KOINA_PID_FILE)))"

## Stop the background Triton/Koina process started by `make koina-up`
koina-down:
	@if [ ! -f $(KOINA_PID_FILE) ]; then \
		echo "[koina-down] no PID file at $(KOINA_PID_FILE); nothing to stop"; \
		exit 0; \
	fi
	@KPID=$$(cat $(KOINA_PID_FILE)); \
	if kill -0 $$KPID 2>/dev/null; then \
		echo "[koina-down] stopping Triton (pid $$KPID)..."; \
		kill $$KPID; \
		while kill -0 $$KPID 2>/dev/null; do sleep 1; done; \
		echo "[koina-down] stopped"; \
	else \
		echo "[koina-down] pid $$KPID is not running"; \
	fi
	@rm -f $(KOINA_PID_FILE)

## Print Koina readiness state and last log lines
koina-status:
	@if curl -fsS $(KOINA_HEALTH_URL) >/dev/null 2>&1; then \
		echo "[koina-status] healthy at $(KOINA_HEALTH_URL)"; \
	else \
		echo "[koina-status] not ready at $(KOINA_HEALTH_URL)"; \
	fi
	@if [ -f $(KOINA_LOG_FILE) ]; then \
		echo "--- last 20 log lines ($(KOINA_LOG_FILE)) ---"; \
		tail -n 20 $(KOINA_LOG_FILE); \
	fi

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
	calibrator.features.retention_time_feature.train_fraction=0.3 \
	$(KOINA_OVERRIDES)

## Run winnow predict with sample data (uses locally trained model from models/new_model)
predict-sample:
	uv run winnow predict \
	calibrator.pretrained_model_name_or_path=models/new_model \
	fdr_control.fdr_threshold=1.0 \
	dataset.spectrum_path_or_directory=examples/example_data/spectra.ipc \
	dataset.predictions_path=examples/example_data/predictions.csv \
	$(KOINA_OVERRIDES)

## Clean output directories (does not delete sample data)
clean:
	rm -rf models/ results/

## Clean outputs and regenerate sample data
clean-all: clean sample-data

#################################################################################
## Runtime benchmarking															#
#################################################################################

.PHONY: benchmark-train-models benchmark benchmark-no-prosit benchmark-scaling

BENCHMARK_TRAIN_SPECTRA := held_out_projects/biological_validation/annotated/dataset-helaqc-annotated-0000-0001.parquet
BENCHMARK_TRAIN_PREDS   := held_out_projects/biological_validation/annotated_predictions/dataset-helaqc-annotated-0000-0001.csv
BENCHMARK_EVAL_SPECTRA  := held_out_projects/biological_validation/raw/dataset-helaqc-raw-0000-0001.parquet
BENCHMARK_EVAL_PREDS    := held_out_projects/biological_validation/raw_predictions/dataset-helaqc-raw-0000-0001.csv

## Train full and no-Prosit calibrators on HeLa QC annotated data
benchmark-train-models:
	uv run winnow train \
	features_path=null \
	dataset.spectrum_path_or_directory=$(BENCHMARK_TRAIN_SPECTRA) \
	dataset.predictions_path=$(BENCHMARK_TRAIN_PREDS) \
	model_output_dir=models/benchmark_model \
	irt_regressor_output_path=null \
	training_history_path=null \
	dataset_output_path=null \
	$(KOINA_OVERRIDES) \
	$(KOINA_FRAGMENT_MATCH_CONSTANTS)
	uv run winnow train \
	features_path=null \
	dataset.spectrum_path_or_directory=$(BENCHMARK_TRAIN_SPECTRA) \
	dataset.predictions_path=$(BENCHMARK_TRAIN_PREDS) \
	model_output_dir=models/benchmark_model_no_prosit \
	irt_regressor_output_path=null \
	training_history_path=null \
	dataset_output_path=null \
	'~calibrator.features.fragment_match_features' \
	'~calibrator.features.retention_time_feature'

## Benchmark predict pipeline on HeLa QC raw data (full and no-Prosit)
benchmark: benchmark-train-models
	uv run python scripts/benchmark_runtime.py \
	--spectrum-path $(BENCHMARK_EVAL_SPECTRA) \
	--predictions-path $(BENCHMARK_EVAL_PREDS) \
	--model-path models/benchmark_model \
	--model-path-no-prosit models/benchmark_model_no_prosit \
	--output-json analysis/benchmark_results.json \
	$(if $(KOINA_SERVER_URL),--koina-url $(KOINA_SERVER_URL)) \
	$(if $(filter false,$(KOINA_SSL)),--koina-ssl false)

## Scaling analysis: run no-Prosit pipeline at 10%, 50%, 100% of data
benchmark-scaling: benchmark-train-models
	uv run python scripts/benchmark_scaling.py \
	--spectrum-path $(BENCHMARK_EVAL_SPECTRA) \
	--predictions-path $(BENCHMARK_EVAL_PREDS) \
	--model-path models/benchmark_model_no_prosit \
	--output-dir analysis

#################################################################################
## General model commands														#
#################################################################################

.PHONY: copy_down_train_dataset compute_train_features

PROJECTS := PXD003868 PXD004325 PXD004424 PXD004452 PXD004467 PXD004536 PXD004732 PXD004947 PXD009449 PXD021013 PXD056559

copy_down_train_dataset:
	$(S3_CP) --recursive s3://winnow-g88rh/revisions/new_datasets/train/ data/train/
	$(S3_CP) --recursive s3://winnow-g88rh/revisions/new_datasets/train_predictions/ data/train_predictions/

compute_train_features: copy_down_train_dataset
	@mkdir -p train_feature_matrices
	@for project in $(PROJECTS); do \
		echo "Computing features for $$project..."; \
		uv run winnow compute-features \
		dataset.spectrum_path_or_directory=data/train/$$project/ \
		dataset.predictions_path=data/train_predictions/$$project.csv \
		training_matrix_output_path=train_feature_matrices/$$project.parquet \
		$(KOINA_OVERRIDES); \
		$(S3_CP) train_feature_matrices/$$project.parquet s3://winnow-g88rh/revisions/new_datasets/train_feature_matrices/$$project.parquet; \
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
	calibrator.seed=42 \
	$(KOINA_OVERRIDES)


#########################################################
## Hyperparameter optimisation (HPO)
#########################################################

.PHONY: hpo hpo-download-data hpo-upload-model

HPO_TRAIN_FEATURES ?= small_train_shuffled/
HPO_VAL_FEATURES   ?= new_val_feature_matrices/PXD010154/
HPO_N_TRIALS       ?= 50
HPO_TIMEOUT        ?= 43200
HPO_CONFIG         ?= scripts/hpo_config.yaml
# RUN_TS is fixed for one make invocation; override on CLI to resume a run.
ifndef RUN_TS
RUN_TS := $(shell date -u +%Y%m%dT%H%M%SZ)
endif
HPO_OUTPUT_DIR     ?= models/hpo_$(RUN_TS)

HPO_MODEL_S3 ?= s3://winnow-g88rh/revisions/new_datasets/models

HPO_TRAIN_S3 ?= s3://winnow-g88rh/revisions/new_datasets/small_train_shuffled
HPO_VAL_S3   ?= s3://winnow-g88rh/revisions/new_datasets/new_val_feature_matrices/PXD010154

## Download feature matrices from S3 for HPO (training + validation).
## Training: all shards under small_train_shuffled/ (~14M PSMs).
## Validation: all shards under new_val_feature_matrices/PXD010154/ (full PXD010154 val).
hpo-download-data:
	mkdir -p $(HPO_TRAIN_FEATURES) $(HPO_VAL_FEATURES)
	$(S3_CP) --recursive $(HPO_TRAIN_S3) $(HPO_TRAIN_FEATURES)
	$(S3_CP) --recursive $(HPO_VAL_S3) $(HPO_VAL_FEATURES)

## Run Optuna HPO for the calibrator on pre-computed feature matrices
hpo: hpo-download-data
	mkdir -p $(HPO_OUTPUT_DIR)
	uv run python scripts/run_hpo.py \
		--train-features-path $(HPO_TRAIN_FEATURES) \
		--val-features-path $(HPO_VAL_FEATURES) \
		--config $(HPO_CONFIG) \
		--n-trials $(HPO_N_TRIALS) \
		--timeout $(HPO_TIMEOUT) \
		--output-dir $(HPO_OUTPUT_DIR) \
		--pruning

## Upload the best HPO model to S3 under models/<timestamp>/
hpo-upload-model:
	@if [ ! -d "$(HPO_OUTPUT_DIR)" ]; then \
		echo "Error: $(HPO_OUTPUT_DIR) does not exist. Run 'make hpo' first."; \
		exit 1; \
	fi
	$(S3_CP) --recursive $(HPO_OUTPUT_DIR) $(RUN_S3)/$(RUN_TS)/model/
	@echo "Model uploaded to $(RUN_S3)/$(RUN_TS)/model/"

#########################################################
## HPO end-to-end pipeline
#########################################################

.PHONY: hpo-pipeline hpo-pipeline-banner download-eval-data eval-annotated eval-raw \
        eval-external-labelled eval-external-unlabelled \
        feature-analysis ablation

S3_CP           := aws s3 cp --no-progress
S3_BASE         ?= s3://winnow-g88rh/revisions/new_datasets
HELD_OUT_S3     ?= $(S3_BASE)/held_out_projects
FASTA_S3        ?= $(S3_BASE)/fasta
RUN_S3          ?= $(S3_BASE)/hpo_runs
PREDS_DIR       ?= predictions/hpo_$(RUN_TS)
EVAL_PLOTS_DIR  ?= analysis/hpo_eval_plots/$(RUN_TS)

EXTERNAL_FULL_PROJECTS := PXD009935 PXD014877
PXD023064_FILES := MSB33410B MSB33411A MSB33659A MSB33663B MSB37876A MSB37878A MSB37880A MSB37884A

# FASTA assignments for raw bio-val proteome annotation (HPO model)
FASTA_RAW_gluc := fasta/human.fasta
FASTA_RAW_helaqc := fasta/human.fasta
FASTA_RAW_herceptin := fasta/herceptin.fasta
FASTA_RAW_immuno := fasta/human.fasta
FASTA_RAW_sbrodae := fasta/Sb_proteome.fasta
FASTA_RAW_snakevenoms := fasta/uniprot-serpentes-2022.05.09.fasta
FASTA_RAW_tplantibodies := fasta/nanobody_library.fasta
FASTA_RAW_woundfluids := fasta/human.fasta

# FASTA assignments for unlabelled external proteome annotation
FASTA_UNLABELLED_PXD009935 := fasta/human.fasta
FASTA_UNLABELLED_PXD014877 := fasta/Celegans.fasta
FASTA_UNLABELLED_PXD023064 := fasta/human.fasta
FASTA_UNLABELLED_Astral := fasta/UP000000625_83333.fasta

## Download all evaluation data from S3
download-eval-data:
	mkdir -p held_out_projects/biological_validation fasta \
	         held_out_projects/lcfm held_out_projects/acfm \
	         astral/labelled astral/full astral/predictions
	$(S3_CP) --recursive $(HELD_OUT_S3)/biological_validation/ held_out_projects/biological_validation/
	$(S3_CP) --recursive $(FASTA_S3)/ fasta/
	@# Full external projects (lcfm + acfm)
	for project in $(EXTERNAL_FULL_PROJECTS); do \
		$(S3_CP) --recursive $(HELD_OUT_S3)/lcfm/$$project/ held_out_projects/lcfm/$$project/; \
		$(S3_CP) --recursive $(HELD_OUT_S3)/lcfm/$${project}_predictions/ held_out_projects/lcfm/$${project}_predictions/; \
		$(S3_CP) --recursive $(HELD_OUT_S3)/acfm/$$project/ held_out_projects/acfm/$$project/; \
		$(S3_CP) --recursive $(HELD_OUT_S3)/acfm/$${project}_predictions/ held_out_projects/acfm/$${project}_predictions/; \
	done
	@# PXD023064 -- only the selected files
	mkdir -p held_out_projects/lcfm/PXD023064 held_out_projects/acfm/PXD023064
	for file in $(PXD023064_FILES); do \
		$(S3_CP) $(HELD_OUT_S3)/lcfm/PXD023064/$$file.parquet held_out_projects/lcfm/PXD023064/$$file.parquet; \
		$(S3_CP) $(HELD_OUT_S3)/acfm/PXD023064/$$file.parquet held_out_projects/acfm/PXD023064/$$file.parquet; \
	done
	$(S3_CP) --recursive $(HELD_OUT_S3)/lcfm/PXD023064_predictions/ held_out_projects/lcfm/PXD023064_predictions/
	$(S3_CP) --recursive $(HELD_OUT_S3)/acfm/PXD023064_predictions/ held_out_projects/acfm/PXD023064_predictions/
	@# Astral
	$(S3_CP) --recursive --exclude "*.ipc" $(S3_BASE)/astral/labelled/ astral/labelled/
	$(S3_CP) --recursive --exclude "*.ipc" $(S3_BASE)/astral/full/ astral/full/
	$(S3_CP) --recursive $(S3_BASE)/astral/predictions/ astral/predictions/

## Evaluate model on annotated bio-val, plot, and upload
eval-annotated:
	mkdir -p $(PREDS_DIR) $(EVAL_PLOTS_DIR)/annotated
	for project in $(BIOLOGICAL_VALIDATION_PROJECTS); do \
		uv run winnow predict \
		dataset.spectrum_path_or_directory=held_out_projects/biological_validation/annotated/dataset-$$project-annotated-0000-0001.parquet \
		dataset.predictions_path=held_out_projects/biological_validation/annotated_predictions/dataset-$$project-annotated-0000-0001.csv \
		calibrator.pretrained_model_name_or_path=$(HPO_OUTPUT_DIR) \
		output_folder=$(PREDS_DIR)/$${project}_annotated/ \
		$(PREDICT_EVAL_OVERRIDES) \
		$(KOINA_FRAGMENT_MATCH_CONSTANTS) \
		fdr_control.fdr_threshold=1.0 \
		fdr_control.confidence_column=calibrated_confidence \
		$(KOINA_OVERRIDES); \
	done
	uv run python scripts/plot_eval_results.py \
		--predictions-root $(PREDS_DIR) \
		--projects "$(BIOLOGICAL_VALIDATION_PROJECTS)" \
		--eval-type annotated \
		--output-dir $(EVAL_PLOTS_DIR)/annotated
	for project in $(BIOLOGICAL_VALIDATION_PROJECTS); do \
		$(S3_CP) --recursive $(PREDS_DIR)/$${project}_annotated/ $(RUN_S3)/$(RUN_TS)/eval_annotated/$${project}_annotated/; \
	done
	$(S3_CP) --recursive $(EVAL_PLOTS_DIR)/annotated/ $(RUN_S3)/$(RUN_TS)/eval_annotated/plots/

## Evaluate model on raw bio-val, annotate proteome hits, plot, and upload
eval-raw:
	mkdir -p $(PREDS_DIR) $(EVAL_PLOTS_DIR)/raw
	for project in $(BIOLOGICAL_VALIDATION_PROJECTS); do \
		uv run winnow predict \
		dataset.spectrum_path_or_directory=held_out_projects/biological_validation/raw/dataset-$$project-raw-0000-0001.parquet \
		dataset.predictions_path=held_out_projects/biological_validation/raw_predictions/dataset-$$project-raw-0000-0001.csv \
		calibrator.pretrained_model_name_or_path=$(HPO_OUTPUT_DIR) \
		output_folder=$(PREDS_DIR)/$${project}_raw/ \
		$(PREDICT_EVAL_OVERRIDES) \
		$(KOINA_FRAGMENT_MATCH_CONSTANTS) \
		fdr_control.fdr_threshold=1.0 \
		fdr_control.confidence_column=calibrated_confidence \
		$(KOINA_OVERRIDES); \
	done
	uv run python scripts/annotate_preds_proteome_hits.py biological_validation_raw \
		--predictions-root $(PREDS_DIR) \
		$(foreach p,$(BIOLOGICAL_VALIDATION_PROJECTS),$(if $(strip $(FASTA_RAW_$(p))),--map $(p)=$(FASTA_RAW_$(p)),))
	uv run python scripts/plot_eval_results.py \
		--predictions-root $(PREDS_DIR) \
		--projects "$(BIOLOGICAL_VALIDATION_PROJECTS)" \
		--eval-type raw \
		--output-dir $(EVAL_PLOTS_DIR)/raw
	for project in $(BIOLOGICAL_VALIDATION_PROJECTS); do \
		$(S3_CP) --recursive $(PREDS_DIR)/$${project}_raw/ $(RUN_S3)/$(RUN_TS)/eval_raw/$${project}_raw/; \
	done
	$(S3_CP) --recursive $(EVAL_PLOTS_DIR)/raw/ $(RUN_S3)/$(RUN_TS)/eval_raw/plots/

## Evaluate model on external labelled datasets, plot, and upload
eval-external-labelled:
	mkdir -p $(PREDS_DIR) $(EVAL_PLOTS_DIR)/external_labelled
	@# Full external projects (have per-row CE/frag columns)
	for project in $(EXTERNAL_FULL_PROJECTS); do \
		uv run winnow predict \
		dataset.spectrum_path_or_directory=held_out_projects/lcfm/$$project/ \
		dataset.predictions_path=held_out_projects/lcfm/$${project}_predictions/$$project.csv \
		calibrator.pretrained_model_name_or_path=$(HPO_OUTPUT_DIR) \
		output_folder=$(PREDS_DIR)/$${project}_labelled/ \
		$(PREDICT_EVAL_OVERRIDES) \
		$(KOINA_FRAGMENT_MATCH_COLUMNS) \
		fdr_control.fdr_threshold=1.0 \
		fdr_control.confidence_column=calibrated_confidence \
		$(KOINA_OVERRIDES); \
	done
	@# PXD023064 -- subset of files (has per-row CE/frag columns)
	uv run winnow predict \
		dataset.spectrum_path_or_directory=held_out_projects/lcfm/PXD023064/ \
		dataset.predictions_path=held_out_projects/lcfm/PXD023064_predictions/PXD023064.csv \
		calibrator.pretrained_model_name_or_path=$(HPO_OUTPUT_DIR) \
		output_folder=$(PREDS_DIR)/PXD023064_labelled/ \
		$(PREDICT_EVAL_OVERRIDES) \
		$(KOINA_FRAGMENT_MATCH_COLUMNS) \
		fdr_control.fdr_threshold=1.0 \
		fdr_control.confidence_column=calibrated_confidence \
		$(KOINA_OVERRIDES)
	@# Astral labelled (has per-row CE/frag columns)
	uv run winnow predict \
		dataset.spectrum_path_or_directory=astral/labelled/ \
		dataset.predictions_path=astral/predictions/astral_labelled.csv \
		calibrator.pretrained_model_name_or_path=$(HPO_OUTPUT_DIR) \
		output_folder=$(PREDS_DIR)/Astral_labelled/ \
		$(PREDICT_EVAL_OVERRIDES) \
		$(KOINA_FRAGMENT_MATCH_COLUMNS) \
		fdr_control.fdr_threshold=1.0 \
		fdr_control.confidence_column=calibrated_confidence \
		$(KOINA_OVERRIDES)
	uv run python scripts/plot_eval_results.py \
		--predictions-root $(PREDS_DIR) \
		--projects "$(EXTERNAL_FULL_PROJECTS) PXD023064 Astral" \
		--eval-type labelled \
		--output-dir $(EVAL_PLOTS_DIR)/external_labelled
	for project in $(EXTERNAL_FULL_PROJECTS) PXD023064 Astral; do \
		$(S3_CP) --recursive $(PREDS_DIR)/$${project}_labelled/ $(RUN_S3)/$(RUN_TS)/eval_external_labelled/$${project}_labelled/; \
	done
	$(S3_CP) --recursive $(EVAL_PLOTS_DIR)/external_labelled/ $(RUN_S3)/$(RUN_TS)/eval_external_labelled/plots/

## Evaluate model on external unlabelled datasets, annotate proteome hits, plot, and upload
eval-external-unlabelled:
	mkdir -p $(PREDS_DIR) $(EVAL_PLOTS_DIR)/external_unlabelled
	@# Full external projects (acfm; have per-row CE/frag columns)
	for project in $(EXTERNAL_FULL_PROJECTS); do \
		uv run winnow predict \
		dataset.spectrum_path_or_directory=held_out_projects/acfm/$$project/ \
		dataset.predictions_path=held_out_projects/acfm/$${project}_predictions/$$project.csv \
		calibrator.pretrained_model_name_or_path=$(HPO_OUTPUT_DIR) \
		output_folder=$(PREDS_DIR)/$${project}_unlabelled/ \
		$(PREDICT_EVAL_OVERRIDES) \
		$(KOINA_FRAGMENT_MATCH_COLUMNS) \
		fdr_control.fdr_threshold=1.0 \
		fdr_control.confidence_column=calibrated_confidence \
		$(KOINA_OVERRIDES); \
	done
	@# PXD023064 -- subset of files (acfm; has per-row CE/frag columns)
	uv run winnow predict \
		dataset.spectrum_path_or_directory=held_out_projects/acfm/PXD023064/ \
		dataset.predictions_path=held_out_projects/acfm/PXD023064_predictions/PXD023064.csv \
		calibrator.pretrained_model_name_or_path=$(HPO_OUTPUT_DIR) \
		output_folder=$(PREDS_DIR)/PXD023064_unlabelled/ \
		$(PREDICT_EVAL_OVERRIDES) \
		$(KOINA_FRAGMENT_MATCH_COLUMNS) \
		fdr_control.fdr_threshold=1.0 \
		fdr_control.confidence_column=calibrated_confidence \
		$(KOINA_OVERRIDES)
	@# Astral full (unlabelled; has per-row CE/frag columns)
	uv run winnow predict \
		dataset.spectrum_path_or_directory=astral/full/ \
		dataset.predictions_path=astral/predictions/astral_full.csv \
		calibrator.pretrained_model_name_or_path=$(HPO_OUTPUT_DIR) \
		output_folder=$(PREDS_DIR)/Astral_unlabelled/ \
		$(PREDICT_EVAL_OVERRIDES) \
		$(KOINA_FRAGMENT_MATCH_COLUMNS) \
		fdr_control.fdr_threshold=1.0 \
		fdr_control.confidence_column=calibrated_confidence \
		$(KOINA_OVERRIDES)
	@# Annotate proteome hits for PXD projects
	uv run python scripts/annotate_preds_proteome_hits.py unlabelled_external \
		--predictions-root $(PREDS_DIR) \
		$(foreach p,$(EXTERNAL_FULL_PROJECTS) PXD023064,$(if $(strip $(FASTA_UNLABELLED_$(p))),--map $(p)=$(FASTA_UNLABELLED_$(p)),))
	@# Annotate proteome hits for Astral with UP000000625_83333.fasta
	uv run python scripts/annotate_preds_proteome_hits.py unlabelled_external \
		--predictions-root $(PREDS_DIR) \
		--map Astral=$(FASTA_UNLABELLED_Astral)
	uv run python scripts/plot_eval_results.py \
		--predictions-root $(PREDS_DIR) \
		--projects "$(EXTERNAL_FULL_PROJECTS) PXD023064 Astral" \
		--eval-type unlabelled \
		--output-dir $(EVAL_PLOTS_DIR)/external_unlabelled
	for project in $(EXTERNAL_FULL_PROJECTS) PXD023064 Astral; do \
		$(S3_CP) --recursive $(PREDS_DIR)/$${project}_unlabelled/ $(RUN_S3)/$(RUN_TS)/eval_external_unlabelled/$${project}_unlabelled/; \
	done
	$(S3_CP) --recursive $(EVAL_PLOTS_DIR)/external_unlabelled/ $(RUN_S3)/$(RUN_TS)/eval_external_unlabelled/plots/

.PHONY: replot-eval-plots-annotated replot-eval-plots-raw \
        replot-eval-plots-external-labelled replot-eval-plots-external-unlabelled \
        replot-eval-plots upload-eval-plots-annotated upload-eval-plots-raw \
        upload-eval-plots-external-labelled upload-eval-plots-external-unlabelled \
        upload-eval-plots

## Replot eval results from local predictions (no predict)
replot-eval-plots-annotated:
	mkdir -p $(EVAL_PLOTS_DIR)/annotated
	uv run python scripts/plot_eval_results.py \
		--predictions-root $(PREDS_DIR) \
		--projects "$(BIOLOGICAL_VALIDATION_PROJECTS)" \
		--eval-type annotated \
		--output-dir $(EVAL_PLOTS_DIR)/annotated

replot-eval-plots-raw:
	mkdir -p $(EVAL_PLOTS_DIR)/raw
	uv run python scripts/plot_eval_results.py \
		--predictions-root $(PREDS_DIR) \
		--projects "$(BIOLOGICAL_VALIDATION_PROJECTS)" \
		--eval-type raw \
		--output-dir $(EVAL_PLOTS_DIR)/raw

replot-eval-plots-external-labelled:
	mkdir -p $(EVAL_PLOTS_DIR)/external_labelled
	uv run python scripts/plot_eval_results.py \
		--predictions-root $(PREDS_DIR) \
		--projects "$(EXTERNAL_FULL_PROJECTS) PXD023064 Astral" \
		--eval-type labelled \
		--output-dir $(EVAL_PLOTS_DIR)/external_labelled

replot-eval-plots-external-unlabelled:
	mkdir -p $(EVAL_PLOTS_DIR)/external_unlabelled
	uv run python scripts/plot_eval_results.py \
		--predictions-root $(PREDS_DIR) \
		--projects "$(EXTERNAL_FULL_PROJECTS) PXD023064 Astral" \
		--eval-type unlabelled \
		--output-dir $(EVAL_PLOTS_DIR)/external_unlabelled

replot-eval-plots: replot-eval-plots-annotated replot-eval-plots-raw \
                   replot-eval-plots-external-labelled replot-eval-plots-external-unlabelled

## Upload replotted eval figures to the S3 eval run prefixes (uses EVAL_TS_*)
upload-eval-plots-annotated:
	$(S3_CP) --recursive $(EVAL_PLOTS_DIR)/annotated/ \
		$(RUN_S3)/$(EVAL_TS_ANNOTATED)/eval_annotated/plots/

upload-eval-plots-raw:
	$(S3_CP) --recursive $(EVAL_PLOTS_DIR)/raw/ \
		$(RUN_S3)/$(EVAL_TS_RAW)/eval_raw/plots/

upload-eval-plots-external-labelled:
	$(S3_CP) --recursive $(EVAL_PLOTS_DIR)/external_labelled/ \
		$(RUN_S3)/$(EVAL_TS_EXTERNAL_LABELLED)/eval_external_labelled/plots/

upload-eval-plots-external-unlabelled:
	$(S3_CP) --recursive $(EVAL_PLOTS_DIR)/external_unlabelled/ \
		$(RUN_S3)/$(EVAL_TS_EXTERNAL_UNLABELLED)/eval_external_unlabelled/plots/

upload-eval-plots: upload-eval-plots-annotated upload-eval-plots-raw \
                   upload-eval-plots-external-labelled upload-eval-plots-external-unlabelled

## Run feature importance analysis on 3 datasets, upload
feature-analysis:
	mkdir -p analysis/feature_analysis/$(RUN_TS)/celegans analysis/feature_analysis/$(RUN_TS)/sbrodae analysis/feature_analysis/$(RUN_TS)/helaqc
	@# C. elegans (PXD014877) -- labelled lcfm, compute features from raw spectra via Koina
	uv run python scripts/analyze_features.py \
		--model-path $(HPO_OUTPUT_DIR) \
		--data-dir . \
		--output-dir analysis/feature_analysis/$(RUN_TS)/celegans \
		--train-spectra held_out_projects/lcfm/PXD014877 \
		--train-preds held_out_projects/lcfm/PXD014877_predictions/PXD014877.csv \
		--test-spectra held_out_projects/lcfm/PXD014877 \
		--test-preds held_out_projects/lcfm/PXD014877_predictions/PXD014877.csv \
		--koina-input-constant collision_energies=27 \
		--koina-input-constant fragmentation_types=HCD \
		--n-background-samples 200 --n-test-samples 500
	@# S. brodae -- annotated bio-val
	uv run python scripts/analyze_features.py \
		--model-path $(HPO_OUTPUT_DIR) \
		--data-dir . \
		--output-dir analysis/feature_analysis/$(RUN_TS)/sbrodae \
		--train-spectra held_out_projects/biological_validation/annotated/dataset-sbrodae-annotated-0000-0001.parquet \
		--train-preds held_out_projects/biological_validation/annotated_predictions/dataset-sbrodae-annotated-0000-0001.csv \
		--test-spectra held_out_projects/biological_validation/annotated/dataset-sbrodae-annotated-0000-0001.parquet \
		--test-preds held_out_projects/biological_validation/annotated_predictions/dataset-sbrodae-annotated-0000-0001.csv \
		--koina-input-constant collision_energies=27 \
		--koina-input-constant fragmentation_types=HCD \
		--n-background-samples 200 --n-test-samples 500
	@# HeLa QC -- annotated bio-val
	uv run python scripts/analyze_features.py \
		--model-path $(HPO_OUTPUT_DIR) \
		--data-dir . \
		--output-dir analysis/feature_analysis/$(RUN_TS)/helaqc \
		--train-spectra held_out_projects/biological_validation/annotated/dataset-helaqc-annotated-0000-0001.parquet \
		--train-preds held_out_projects/biological_validation/annotated_predictions/dataset-helaqc-annotated-0000-0001.csv \
		--test-spectra held_out_projects/biological_validation/annotated/dataset-helaqc-annotated-0000-0001.parquet \
		--test-preds held_out_projects/biological_validation/annotated_predictions/dataset-helaqc-annotated-0000-0001.csv \
		--koina-input-constant collision_energies=27 \
		--koina-input-constant fragmentation_types=HCD \
		--n-background-samples 200 --n-test-samples 500
	$(S3_CP) --recursive analysis/feature_analysis/$(RUN_TS)/ $(RUN_S3)/$(RUN_TS)/feature_analysis/

## Run feature ablation study on 4 datasets, upload
ablation:
	uv run python scripts/run_feature_ablations.py \
		--train-features $(HPO_TRAIN_FEATURES) \
		--val-features $(HPO_VAL_FEATURES) \
		--hyperparams-from-model $(HPO_OUTPUT_DIR) \
		--output-dir analysis/hpo_ablation/$(RUN_TS) \
		--astral-spectra astral/labelled \
		--astral-predictions astral/predictions/astral_labelled.csv \
		--plot-format both \
		$(if $(KOINA_SERVER_URL),--koina-url $(KOINA_SERVER_URL)) \
		$(if $(filter false,$(KOINA_SSL)),--no-koina-ssl)
	$(S3_CP) --recursive analysis/hpo_ablation/$(RUN_TS)/ $(RUN_S3)/$(RUN_TS)/ablation/

## Print run identifiers then run the full HPO pipeline
hpo-pipeline-banner:
	@echo "=== HPO pipeline ==="
	@echo "  RUN_TS=$(RUN_TS)"
	@echo "  HPO_OUTPUT_DIR=$(HPO_OUTPUT_DIR)"
	@echo "  PREDS_DIR=$(PREDS_DIR)"
	@echo "  S3 prefix=$(RUN_S3)/$(RUN_TS)/"

## Full HPO pipeline: tune, evaluate, analyse, upload
hpo-pipeline: hpo-pipeline-banner hpo-download-data hpo hpo-upload-model download-eval-data \
              eval-annotated eval-raw eval-external-labelled \
              eval-external-unlabelled feature-analysis ablation

#########################################################
## Re-run eval pipeline with a saved HPO model from S3
#########################################################

.PHONY: hpo-download-model hpo-reeval hpo-reeval-annotated hpo-reeval-raw \
        hpo-reeval-external-labelled hpo-reeval-external-unlabelled \
        hpo-reeval-feature-analysis hpo-reeval-ablation \
        hpo-replot-eval hpo-replot-eval-annotated hpo-replot-eval-raw \
        hpo-replot-eval-external-labelled hpo-replot-eval-external-unlabelled

# S3 path to the HPO run whose model you want to evaluate.
# Override on the command line:
#   make hpo-reeval HPO_RUN_TS=20260514T012035Z
HPO_RUN_TS ?= 20260514T012035Z
HPO_MODEL_S3_PATH ?= $(RUN_S3)/$(HPO_RUN_TS)/model

## Download a saved HPO model from S3
hpo-download-model:
	mkdir -p $(HPO_OUTPUT_DIR)
	$(S3_CP) --recursive $(HPO_MODEL_S3_PATH)/ $(HPO_OUTPUT_DIR)/

## Re-run the full eval pipeline using a previously saved HPO model
hpo-reeval: hpo-download-model download-eval-data \
            eval-annotated eval-raw eval-external-labelled \
            eval-external-unlabelled

## Re-run only annotated bio-val evaluation
hpo-reeval-annotated: hpo-download-model download-eval-data eval-annotated

## Re-run only raw bio-val evaluation
hpo-reeval-raw: hpo-download-model download-eval-data eval-raw

## Re-run only external labelled evaluation
hpo-reeval-external-labelled: hpo-download-model download-eval-data eval-external-labelled

## Re-run only external unlabelled evaluation
hpo-reeval-external-unlabelled: hpo-download-model download-eval-data eval-external-unlabelled

## Re-run feature analysis with a saved HPO model
hpo-reeval-feature-analysis: hpo-download-model download-eval-data feature-analysis

## Re-run ablation study with a saved HPO model
hpo-reeval-ablation: hpo-download-model hpo-download-data download-eval-data ablation

# Replot eval figures from saved S3 predictions and upload to each eval run prefix.
# Override EVAL_TS_* to match the hpo_runs/ folders on S3, e.g.:
#   make hpo-replot-eval \
#     EVAL_TS_ANNOTATED=20260514T155345Z \
#     EVAL_TS_RAW=20260514T160855Z \
#     EVAL_TS_EXTERNAL_LABELLED=20260514T191009Z \
#     EVAL_TS_EXTERNAL_UNLABELLED=20260514T204748Z \
#     S3_CP='aws s3 cp --no-progress --profile winnow'
## Download eval preds from S3, replot, and upload plots (no predict)
hpo-replot-eval: download-eval-preds replot-eval-plots upload-eval-plots

## Replot and upload only annotated bio-val eval
hpo-replot-eval-annotated: download-eval-preds replot-eval-plots-annotated upload-eval-plots-annotated

## Replot and upload only raw bio-val eval
hpo-replot-eval-raw: download-eval-preds replot-eval-plots-raw upload-eval-plots-raw

## Replot and upload only external labelled eval
hpo-replot-eval-external-labelled: download-eval-preds \
	replot-eval-plots-external-labelled upload-eval-plots-external-labelled

## Replot and upload only external unlabelled eval
hpo-replot-eval-external-unlabelled: download-eval-preds \
	replot-eval-plots-external-unlabelled upload-eval-plots-external-unlabelled

ABLATION_OUTPUT_DIR ?= analysis/hpo_ablation
ABLATION_S3_PATH   ?= $(RUN_S3)/$(HPO_RUN_TS)/ablation

.PHONY: ablation-download ablation-replot

## Download saved ablation models and cached eval features from S3
ablation-download:
	mkdir -p $(ABLATION_OUTPUT_DIR)
	$(S3_CP) --recursive $(ABLATION_S3_PATH)/ $(ABLATION_OUTPUT_DIR)/

## Re-run evaluation and replot from saved ablation models (no training, no feature compute)
ablation-replot: ablation-download
	uv run python scripts/run_feature_ablations.py \
		--output-dir $(ABLATION_OUTPUT_DIR) \
		--skip-training \
		--skip-feature-compute \
		--plot-format both
	$(S3_CP) --recursive $(ABLATION_OUTPUT_DIR)/ $(ABLATION_S3_PATH)/

#########################################################
## Evaluate general model commands
#########################################################

.PHONY: evaluate_general_model_annotated_biological_validation evaluate_general_model_raw_biological_validation evaluate_general_model_labelled_external_datasets evaluate_general_model_unlabelled_external_datasets annotate_preds_proteome_hits generalisation_analysis generalisation_heatmaps analyze_features analyze_upscored_fps analyze_fdr_overlap

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
		$(PREDICT_EVAL_OVERRIDES) \
		$(KOINA_FRAGMENT_MATCH_CONSTANTS) \
		fdr_control.fdr_threshold=1.0 \
		fdr_control.confidence_column=calibrated_confidence \
		$(KOINA_OVERRIDES); \
	done

evaluate_general_model_raw_biological_validation:
	for project in $(BIOLOGICAL_VALIDATION_PROJECTS); do \
		uv run winnow predict \
		dataset.spectrum_path_or_directory=held_out_projects/biological_validation/raw/dataset-$$project-raw-0000-0001.parquet \
		dataset.predictions_path=held_out_projects/biological_validation/raw_predictions/dataset-$$project-raw-0000-0001.csv \
		calibrator.pretrained_model_name_or_path=general_model \
		output_folder=predictions/general_model/$${project}_raw/ \
		$(PREDICT_EVAL_OVERRIDES) \
		$(KOINA_FRAGMENT_MATCH_CONSTANTS) \
		fdr_control.fdr_threshold=1.0 \
		fdr_control.confidence_column=calibrated_confidence \
		$(KOINA_OVERRIDES); \
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
		$(PREDICT_EVAL_OVERRIDES) \
		$(KOINA_FRAGMENT_MATCH_COLUMNS) \
		fdr_control.fdr_threshold=1.0 \
		fdr_control.confidence_column=calibrated_confidence \
		$(KOINA_OVERRIDES); \
	done

evaluate_general_model_unlabelled_external_datasets:
	for project in $(EXTERNAL_DATASETS); do \
		uv run winnow predict \
		dataset.spectrum_path_or_directory=held_out_projects/acfm/$$project/ \
		dataset.predictions_path=held_out_projects/acfm/$${project}_predictions/$$project.csv \
		calibrator.pretrained_model_name_or_path=general_model \
		output_folder=predictions/general_model/$${project}_unlabelled/ \
		$(PREDICT_EVAL_OVERRIDES) \
		$(KOINA_FRAGMENT_MATCH_COLUMNS) \
		fdr_control.fdr_threshold=1.0 \
		fdr_control.confidence_column=calibrated_confidence \
		$(KOINA_OVERRIDES); \
	done
	uv run python scripts/annotate_preds_proteome_hits.py unlabelled_external \
		$(foreach p,$(EXTERNAL_DATASETS),$(if $(strip $(PROTEOME_FASTA_unlabelled_$(p))),--map $(p)=$(PROTEOME_FASTA_unlabelled_$(p)),))

#########################################################
## Calibrator generalisation analysis
#########################################################

## Train per-dataset calibrators and cross-evaluate on biological validation datasets
generalisation_analysis:
	uv run python scripts/evaluate_calibrator_generalisation.py \
		--data-dir held_out_projects/biological_validation/annotated \
		--predictions-dir held_out_projects/biological_validation/annotated_predictions \
		--model-output-dir analysis/generalisation \
		--results-output-dir analysis/generalisation \
		$(if $(KOINA_SERVER_URL),--koina-server-url $(KOINA_SERVER_URL)) \
		$(if $(filter false,$(KOINA_SSL)),--no-koina-ssl)

## Plot PR-AUC heatmaps from generalisation results (runs generalisation_analysis first)
generalisation_heatmaps: generalisation_analysis
	uv run python scripts/plot_calibrator_generalisation_heatmap.py \
		--results-path analysis/generalisation/calibrator_generalisation_results.csv \
		--output-dir analysis/generalisation/plots

#########################################################
## Feature importance analysis
#########################################################

# Override these on the command line, e.g.:
#   make analyze_features FEATURE_ANALYSIS_MODEL_PATH=general_model FEATURE_ANALYSIS_DATA_DIR=data/
FEATURE_ANALYSIS_MODEL_PATH ?=
FEATURE_ANALYSIS_DATA_DIR ?=

## Analyze feature importance and SHAP values for a pretrained calibrator
analyze_features:
	@if [ -z "$(FEATURE_ANALYSIS_MODEL_PATH)" ] || [ -z "$(FEATURE_ANALYSIS_DATA_DIR)" ]; then \
		echo "Error: set FEATURE_ANALYSIS_MODEL_PATH and FEATURE_ANALYSIS_DATA_DIR"; \
		echo "  make analyze_features FEATURE_ANALYSIS_MODEL_PATH=general_model FEATURE_ANALYSIS_DATA_DIR=data/"; \
		exit 1; \
	fi
	uv run python scripts/analyze_features.py \
		--model-path $(FEATURE_ANALYSIS_MODEL_PATH) \
		--data-dir $(FEATURE_ANALYSIS_DATA_DIR) \
		--output-dir analysis/feature_analysis

#########################################################
## Reviewer analyses: up-scored FPs and post-FDR overlap
#########################################################

.PHONY: download-eval-preds analyze-upscored-fps analyze-fdr-overlap \
        upload-reviewer-analyses reviewer-analyses \
        analyze_upscored_fps analyze_fdr_overlap

# S3 timestamps for each evaluation stage (override on CLI if needed)
EVAL_TS_ANNOTATED          ?= 20260514T155345Z
EVAL_TS_RAW                ?= 20260514T160855Z
EVAL_TS_EXTERNAL_LABELLED  ?= 20260514T191009Z
EVAL_TS_EXTERNAL_UNLABELLED ?= 20260514T204748Z

ANALYSIS_PREDS_DIR ?= $(PREDS_DIR)

## Download winnow predict outputs from S3 eval runs into $(ANALYSIS_PREDS_DIR).
## Downloads preds_and_fdr_metrics.csv from all evals, and metadata.csv only from labelled evals.
download-eval-preds:
	@echo "=== Downloading eval predictions ==="
	@# Annotated bio-val (labelled) — preds + metadata
	for project in $(BIOLOGICAL_VALIDATION_PROJECTS); do \
		mkdir -p $(ANALYSIS_PREDS_DIR)/$${project}_annotated; \
		$(S3_CP) $(RUN_S3)/$(EVAL_TS_ANNOTATED)/eval_annotated/$${project}_annotated/preds_and_fdr_metrics.csv \
			$(ANALYSIS_PREDS_DIR)/$${project}_annotated/preds_and_fdr_metrics.csv; \
		$(S3_CP) $(RUN_S3)/$(EVAL_TS_ANNOTATED)/eval_annotated/$${project}_annotated/metadata.csv \
			$(ANALYSIS_PREDS_DIR)/$${project}_annotated/metadata.csv; \
	done
	@# Raw bio-val (unlabelled) — preds only
	for project in $(BIOLOGICAL_VALIDATION_PROJECTS); do \
		mkdir -p $(ANALYSIS_PREDS_DIR)/$${project}_raw; \
		$(S3_CP) $(RUN_S3)/$(EVAL_TS_RAW)/eval_raw/$${project}_raw/preds_and_fdr_metrics.csv \
			$(ANALYSIS_PREDS_DIR)/$${project}_raw/preds_and_fdr_metrics.csv; \
	done
	@# External labelled — preds + metadata
	for project in $(EXTERNAL_FULL_PROJECTS) PXD023064 Astral; do \
		mkdir -p $(ANALYSIS_PREDS_DIR)/$${project}_labelled; \
		$(S3_CP) $(RUN_S3)/$(EVAL_TS_EXTERNAL_LABELLED)/eval_external_labelled/$${project}_labelled/preds_and_fdr_metrics.csv \
			$(ANALYSIS_PREDS_DIR)/$${project}_labelled/preds_and_fdr_metrics.csv; \
		$(S3_CP) $(RUN_S3)/$(EVAL_TS_EXTERNAL_LABELLED)/eval_external_labelled/$${project}_labelled/metadata.csv \
			$(ANALYSIS_PREDS_DIR)/$${project}_labelled/metadata.csv; \
	done
	@# External unlabelled — preds only
	for project in $(EXTERNAL_FULL_PROJECTS) PXD023064 Astral; do \
		mkdir -p $(ANALYSIS_PREDS_DIR)/$${project}_unlabelled; \
		$(S3_CP) $(RUN_S3)/$(EVAL_TS_EXTERNAL_UNLABELLED)/eval_external_unlabelled/$${project}_unlabelled/preds_and_fdr_metrics.csv \
			$(ANALYSIS_PREDS_DIR)/$${project}_unlabelled/preds_and_fdr_metrics.csv; \
	done
	@echo "=== download-eval-preds complete ==="

## Characterise up-scored FPs from downloaded eval predictions
analyze-upscored-fps: download-eval-preds
	uv run python scripts/analyze_upscored_fps.py \
		--predictions-root $(ANALYSIS_PREDS_DIR) \
		--output-dir analysis/upscored_fps

## Post-FDR overlap from downloaded eval predictions (labelled + unlabelled)
analyze-fdr-overlap: download-eval-preds
	uv run python scripts/analyze_fdr_overlap.py \
		--predictions-root $(ANALYSIS_PREDS_DIR) \
		--output-dir analysis/fdr_overlap \
		--include-unlabelled

## Upload reviewer analysis results to S3
upload-reviewer-analyses:
	$(S3_CP) --recursive analysis/upscored_fps/ $(RUN_S3)/$(HPO_RUN_TS)/reviewer_analyses/upscored_fps/
	$(S3_CP) --recursive analysis/fdr_overlap/  $(RUN_S3)/$(HPO_RUN_TS)/reviewer_analyses/fdr_overlap/

## Download eval predictions, run both reviewer analyses, and upload results
reviewer-analyses: analyze-upscored-fps analyze-fdr-overlap upload-reviewer-analyses

## Characterise false positives that calibration up-scores into high-confidence regions
analyze_upscored_fps:
	uv run python scripts/analyze_upscored_fps.py \
		--predictions-root predictions/general_model \
		--output-dir analysis/upscored_fps

## Post-FDR overlap: Winnow-filtered identifications vs database-search ground truth
analyze_fdr_overlap:
	uv run python scripts/analyze_fdr_overlap.py \
		--predictions-root predictions/general_model \
		--output-dir analysis/fdr_overlap \
		--include-unlabelled

#########################################################
## Novelty analysis (reviewer response)
#########################################################

.PHONY: download-pxd004732 download-novelty-preds replot-novelty-plots \
        replot-novelty upload-novelty-plots upload-novelty-replot \
        eval-novelty analyze-novelty upload-novelty-analysis novelty-pipeline

PXD004732_S3 := $(S3_BASE)/PXD004732
NOVELTY_ANALYSIS_DIR ?= analysis/novelty
# Local folder for novelty predict outputs; defaults to match HPO_RUN_TS on S3.
NOVELTY_PREDS_DIR ?= predictions/hpo_$(HPO_RUN_TS)

## Download PXD004732 lcfm/acfm data and predictions from S3
download-pxd004732:
	mkdir -p held_out_projects/PXD004732/lcfm \
	         held_out_projects/PXD004732/acfm \
	         held_out_projects/PXD004732/predictions
	$(S3_CP) --recursive $(PXD004732_S3)/lcfm/ held_out_projects/PXD004732/lcfm/ --exclude "*.parque"
	$(S3_CP) --recursive $(PXD004732_S3)/acfm/ held_out_projects/PXD004732/acfm/ --exclude "*.parque"
	$(S3_CP) $(PXD004732_S3)/lcfm/PXD004732.csv held_out_projects/PXD004732/predictions/PXD004732_lcfm.csv
	$(S3_CP) $(PXD004732_S3)/acfm/predictions/PXD004732.csv held_out_projects/PXD004732/predictions/PXD004732_acfm.csv

## Run winnow predict on GluC raw + PXD004732 lcfm + PXD004732 acfm
eval-novelty: download-pxd004732 download-eval-data hpo-download-model
	@# GluC raw (biological validation, no per-row CE/frag columns)
	uv run winnow predict \
		dataset.spectrum_path_or_directory=held_out_projects/biological_validation/raw/dataset-gluc-raw-0000-0001.parquet \
		dataset.predictions_path=held_out_projects/biological_validation/raw_predictions/dataset-gluc-raw-0000-0001.csv \
		calibrator.pretrained_model_name_or_path=$(HPO_OUTPUT_DIR) \
		output_folder=$(PREDS_DIR)/gluc_raw/ \
		$(PREDICT_EVAL_OVERRIDES) \
		$(KOINA_FRAGMENT_MATCH_CONSTANTS) \
		fdr_control.fdr_threshold=1.0 \
		fdr_control.confidence_column=calibrated_confidence \
		$(KOINA_OVERRIDES)
	@# PXD004732 lcfm (labelled, has per-row CE/frag columns)
	uv run winnow predict \
		dataset.spectrum_path_or_directory=held_out_projects/PXD004732/lcfm/ \
		dataset.predictions_path=held_out_projects/PXD004732/predictions/PXD004732_lcfm.csv \
		calibrator.pretrained_model_name_or_path=$(HPO_OUTPUT_DIR) \
		output_folder=$(PREDS_DIR)/PXD004732_labelled/ \
		$(PREDICT_EVAL_OVERRIDES) \
		$(KOINA_FRAGMENT_MATCH_COLUMNS) \
		fdr_control.fdr_threshold=1.0 \
		fdr_control.confidence_column=calibrated_confidence \
		$(KOINA_OVERRIDES)
	@# PXD004732 acfm (unlabelled, has per-row CE/frag columns)
	uv run winnow predict \
		dataset.spectrum_path_or_directory=held_out_projects/PXD004732/acfm/ \
		dataset.predictions_path=held_out_projects/PXD004732/predictions/PXD004732_acfm.csv \
		calibrator.pretrained_model_name_or_path=$(HPO_OUTPUT_DIR) \
		output_folder=$(PREDS_DIR)/PXD004732_unlabelled/ \
		$(PREDICT_EVAL_OVERRIDES) \
		$(KOINA_FRAGMENT_MATCH_COLUMNS) \
		fdr_control.fdr_threshold=1.0 \
		fdr_control.confidence_column=calibrated_confidence \
		$(KOINA_OVERRIDES)

## Download saved novelty predict outputs from S3 (no winnow predict).
## Override HPO_RUN_TS to match the hpo_runs/ folder on S3, e.g.:
#   make download-novelty-preds HPO_RUN_TS=20260514T012035Z \
#     S3_CP='aws s3 cp --no-progress --profile winnow'
download-novelty-preds:
	@echo "=== Downloading novelty predictions from $(RUN_S3)/$(HPO_RUN_TS)/eval_novelty/ ==="
	mkdir -p $(NOVELTY_PREDS_DIR)/gluc_raw \
	         $(NOVELTY_PREDS_DIR)/PXD004732_labelled \
	         $(NOVELTY_PREDS_DIR)/PXD004732_unlabelled \
	         fasta
	$(S3_CP) --recursive $(RUN_S3)/$(HPO_RUN_TS)/eval_novelty/gluc_raw/ \
		$(NOVELTY_PREDS_DIR)/gluc_raw/
	$(S3_CP) --recursive $(RUN_S3)/$(HPO_RUN_TS)/eval_novelty/PXD004732_labelled/ \
		$(NOVELTY_PREDS_DIR)/PXD004732_labelled/
	$(S3_CP) --recursive $(RUN_S3)/$(HPO_RUN_TS)/eval_novelty/PXD004732_unlabelled/ \
		$(NOVELTY_PREDS_DIR)/PXD004732_unlabelled/
	$(S3_CP) $(FASTA_S3)/human.fasta fasta/human.fasta
	@echo "=== download-novelty-preds complete ==="

## Replot novelty analysis from local predictions (no predict)
replot-novelty-plots:
	mkdir -p $(NOVELTY_ANALYSIS_DIR)/gluc $(NOVELTY_ANALYSIS_DIR)/proteometools
	uv run python scripts/analyze_novelty.py gluc \
		--predictions-dir $(NOVELTY_PREDS_DIR)/gluc_raw \
		--fasta fasta/human.fasta \
		--output-dir $(NOVELTY_ANALYSIS_DIR)/gluc
	uv run python scripts/analyze_novelty.py proteometools \
		--lcfm-predictions-dir $(NOVELTY_PREDS_DIR)/PXD004732_labelled \
		--acfm-predictions-dir $(NOVELTY_PREDS_DIR)/PXD004732_unlabelled \
		--output-dir $(NOVELTY_ANALYSIS_DIR)/proteometools
	uv run python scripts/analyze_novelty.py summary \
		--gluc-dir $(NOVELTY_ANALYSIS_DIR)/gluc \
		--proteometools-dir $(NOVELTY_ANALYSIS_DIR)/proteometools \
		--output-dir $(NOVELTY_ANALYSIS_DIR)

## Download novelty preds from S3 and replot (no predict, no upload)
replot-novelty: download-novelty-preds replot-novelty-plots

## Upload replotted novelty figures/tables only
upload-novelty-plots:
	$(S3_CP) --recursive $(NOVELTY_ANALYSIS_DIR)/ \
		$(RUN_S3)/$(HPO_RUN_TS)/reviewer_analyses/novelty/

## Download, replot, and upload novelty analysis artefacts
upload-novelty-replot: replot-novelty upload-novelty-plots

## Run novelty analysis (GluC + ProteomeTools) and produce plots/tables
analyze-novelty: eval-novelty
	$(MAKE) replot-novelty-plots NOVELTY_PREDS_DIR=$(PREDS_DIR)

## Upload novelty analysis artefacts to S3
upload-novelty-analysis:
	$(S3_CP) --recursive $(NOVELTY_ANALYSIS_DIR)/ $(RUN_S3)/$(HPO_RUN_TS)/reviewer_analyses/novelty/
	$(S3_CP) --recursive $(PREDS_DIR)/gluc_raw/ $(RUN_S3)/$(HPO_RUN_TS)/eval_novelty/gluc_raw/
	$(S3_CP) --recursive $(PREDS_DIR)/PXD004732_labelled/ $(RUN_S3)/$(HPO_RUN_TS)/eval_novelty/PXD004732_labelled/
	$(S3_CP) --recursive $(PREDS_DIR)/PXD004732_unlabelled/ $(RUN_S3)/$(HPO_RUN_TS)/eval_novelty/PXD004732_unlabelled/

## Full novelty pipeline: download, predict, analyze, upload
novelty-pipeline: analyze-novelty upload-novelty-analysis

#########################################################
## Compute features for new training dataset
#########################################################

NEW_TRAIN_S3 = s3://winnow-g88rh/revisions/new_datasets/new_training_set
NEW_TRAIN_FEATURES_S3 = s3://winnow-g88rh/revisions/new_datasets/new_training_set_feature_matrices

NEW_TRAIN_PROJECTS = PXD000865 PXD004452 PXD006939 PXD013868 PXD019483

# Projects whose experiments are shuffled across parquet shards and must be
# combined into a single file before feature computation.
SHUFFLED_SHARD_PROJECTS = acpt massivekb

.PHONY: compute_features_new_training_dataset \
	$(addprefix compute_features_,$(NEW_TRAIN_PROJECTS))

# Template for shuffled-shard projects: download, combine shards, compute
# features on the single combined parquet, then upload.
define compute_features_shuffled_project_template
compute_features_$(1):
	mkdir -p data/train/$(1)/
	mkdir -p data/train_predictions/
	$(S3_CP) $(NEW_TRAIN_S3)/$(1)/ data/train/$(1)/ --recursive
	$(S3_CP) $(NEW_TRAIN_S3)/predictions/$(1).csv data/train_predictions/$(1).csv
	@echo "Combining parquet shards in data/train/$(1)/ -> data/train/$(1).parquet"
	uv run python -c "import polars as pl; from pathlib import Path; d = Path('data/train/$(1)'); parts = sorted(d.glob('*.parquet')); print(f'  Found {len(parts)} shard(s), {sum(pl.scan_parquet(p).select(pl.len()).collect().item() for p in parts):,} rows total'); pl.concat([pl.read_parquet(p) for p in parts]).write_parquet(d.parent / '$(1).parquet'); print(f'  Written to data/train/$(1).parquet')"
	uv run winnow compute-features \
		dataset.spectrum_path_or_directory=data/train/$(1).parquet \
		dataset.predictions_path=data/train_predictions/$(1).csv \
		training_matrix_output_path=train_feature_matrices/$(1)/training_matrix.parquet \
		koina.server_url=localhost:8500 koina.ssl=false
	$(S3_CP) train_feature_matrices/$(1)/ $(NEW_TRAIN_FEATURES_S3)/$(1)/ --recursive
endef

# Template for normal projects: download directory, compute features per-file.
define compute_features_project_template
compute_features_$(1):
	mkdir -p data/train/$(1)/
	mkdir -p data/train_predictions/
	$(S3_CP) $(NEW_TRAIN_S3)/$(1)/ data/train/$(1)/ --recursive
	$(S3_CP) $(NEW_TRAIN_S3)/predictions/$(1).csv data/train_predictions/$(1).csv
	uv run winnow compute-features \
		dataset.spectrum_path_or_directory=data/train/$(1)/ \
		dataset.predictions_path=data/train_predictions/$(1).csv \
		training_matrix_output_path=train_feature_matrices/$(1)/training_matrix.parquet \
		koina.server_url=localhost:8500 koina.ssl=false
	$(S3_CP) train_feature_matrices/$(1)/ $(NEW_TRAIN_FEATURES_S3)/$(1)/ --recursive
endef

$(foreach proj,$(SHUFFLED_SHARD_PROJECTS),$(eval $(call compute_features_shuffled_project_template,$(proj))))
$(foreach proj,$(filter-out $(SHUFFLED_SHARD_PROJECTS),$(NEW_TRAIN_PROJECTS)),$(eval $(call compute_features_project_template,$(proj))))

compute_features_new_training_dataset:
	$(foreach proj,$(NEW_TRAIN_PROJECTS),$(MAKE) compute_features_$(proj) && ) true

#########################################################
## Compute features for large, batched datasets
#########################################################

PXD000561_BATCHES = batch_1 sub_1 sub_2
PXD010154_BATCHES = batch_1 batch_2 batch_3 batch_4
PXD024364_BATCHES = batch_5 sub_1 sub_2 sub_3 sub_4 sub_5 sub_6 sub_7 sub_8 sub_9 sub_10 sub_11 sub_12 sub_13 sub_14

NEW_TRAIN_BATCH_TARGETS = \
	$(addprefix compute_features_PXD000561_,$(PXD000561_BATCHES)) \
	$(addprefix compute_features_PXD010154_,$(PXD010154_BATCHES)) \
	$(addprefix compute_features_PXD024364_,$(PXD024364_BATCHES))

.PHONY: compute_features_new_training_dataset_batched \
	compute_features_PXD000561_batched compute_features_PXD010154_batched compute_features_PXD024364_batched \
	$(NEW_TRAIN_BATCH_TARGETS)

define compute_features_batch_template
compute_features_$(1)_$(2):
	mkdir -p data/train/$(1)/$(2)/
	mkdir -p data/train_predictions/
	$(S3_CP) $(NEW_TRAIN_S3)/$(1)/$(2)/ data/train/$(1)/$(2)/ --recursive
	$(S3_CP) $(NEW_TRAIN_S3)/predictions/$(1)_$(2).csv data/train_predictions/$(1)_$(2).csv
	uv run winnow compute-features \
		dataset.spectrum_path_or_directory=data/train/$(1)/$(2)/ \
		dataset.predictions_path=data/train_predictions/$(1)_$(2).csv \
		training_matrix_output_path=train_feature_matrices/$(1)/$(2)/training_matrix.parquet \
		koina.server_url=localhost:8500 koina.ssl=false
	$(S3_CP) train_feature_matrices/$(1)/$(2)/ $(NEW_TRAIN_FEATURES_S3)/$(1)/$(2)/ --recursive
	rm -rf data/train/$(1)/$(2)/
	rm -f data/train_predictions/$(1)_$(2).csv
endef

$(foreach batch,$(PXD000561_BATCHES),$(eval $(call compute_features_batch_template,PXD000561,$(batch))))
$(foreach batch,$(PXD010154_BATCHES),$(eval $(call compute_features_batch_template,PXD010154,$(batch))))
$(foreach batch,$(PXD024364_BATCHES),$(eval $(call compute_features_batch_template,PXD024364,$(batch))))

compute_features_PXD000561_batched:
	$(foreach b,$(PXD000561_BATCHES),$(MAKE) compute_features_PXD000561_$(b) && ) true

compute_features_PXD010154_batched:
	$(foreach b,$(PXD010154_BATCHES),$(MAKE) compute_features_PXD010154_$(b) && ) true

compute_features_PXD024364_batched:
	$(foreach b,$(PXD024364_BATCHES),$(MAKE) compute_features_PXD024364_$(b) && ) true

PXD024364_BATCHES_PART1 = batch_5 sub_1 sub_2 sub_3 sub_4 sub_5 sub_6
PXD024364_BATCHES_PART2 = sub_7 sub_8 sub_9 sub_10 sub_11 sub_12 sub_13 sub_14

.PHONY: compute_features_PXD024364_batched_part1 compute_features_PXD024364_batched_part2

compute_features_PXD024364_batched_part1:
	$(foreach b,$(PXD024364_BATCHES_PART1),$(MAKE) compute_features_PXD024364_$(b) && ) true

compute_features_PXD024364_batched_part2:
	$(foreach b,$(PXD024364_BATCHES_PART2),$(MAKE) compute_features_PXD024364_$(b) && ) true

compute_features_new_training_dataset_batched: compute_features_PXD000561_batched compute_features_PXD010154_batched compute_features_PXD024364_batched

#########################################################
## Per-model analysis: train / predict / plot
## (instanovo, casanovo, primenovo × helaqc, celegans)
#########################################################

# ── FASTA per organism ─────────────────────────────────
ANALYSIS_FASTA_helaqc     := fasta/human.fasta
ANALYSIS_FASTA_celegans    := fasta/Celegans.fasta
ANALYSIS_FASTA_pxd019483  := fasta/human.fasta
ANALYSIS_FASTA_sbrodae     := fasta/Sb_proteome.fasta

# ── External model repo roots ──────────────────────────
# Locally these point to sibling repos; on the cluster they live under
# analysis_data/ (populated by `make download-analysis-data`).
CASANOVO_ROOT  ?= /home/j-daniel/repos/casanovo
PRIMENOVO_ROOT ?= /home/j-daniel/repos/pi-PrimeNovo

# ── Result directory roots ─────────────────────────────
ANALYSIS_MODELS  := models
ANALYSIS_RESULTS := results

# Upload preds + metadata to $(ANALYSIS_S3)/results/<dataset>/ after each predict.
define upload_analysis_predictions
	@$(MAKE) --no-print-directory upload-analysis-predictions-dir \
		OUTPUT_DIR=$(1)
endef

.PHONY: upload-analysis-predictions-dir
upload-analysis-predictions-dir:
	@out="$(OUTPUT_DIR)"; \
	if [ ! -f "$$out/preds_and_fdr_metrics.csv" ]; then \
		echo "WARN: no preds_and_fdr_metrics.csv in $$out, skipping upload"; \
		exit 0; \
	fi; \
	name=$$(basename "$$out"); \
	dest="$(ANALYSIS_S3)/results/$$name"; \
	echo "Uploading predictions $$name -> $$dest"; \
	$(S3_CP) "$$out/preds_and_fdr_metrics.csv" "$$dest/"; \
	if [ -f "$$out/metadata.csv" ]; then \
		$(S3_CP) "$$out/metadata.csv" "$$dest/"; \
	fi

# ── Common Hydra overrides for per-model analysis ──────
# These datasets use constant CE/frag, not per-row columns.
ANALYSIS_KOINA_OVERRIDES = $(KOINA_OVERRIDES) $(KOINA_FRAGMENT_MATCH_CONSTANTS)
ANALYSIS_PREDICT_OVERRIDES = $(PREDICT_EVAL_OVERRIDES)

# Split parquets predicted from MGF exports use spectrum_id = {mgf_stem}:{index}.
# Re-key parquet spectrum_id to match before train/predict (see scripts/rekey_split_parquet_spectrum_ids.py).
.PHONY: rekey-split-parquet-spectrum-ids \
	rekey-pxd019483-split-parquet-spectrum-ids \
	rekey-sbrodae-split-parquet-spectrum-ids \
	rekey-celegans-split-parquet-spectrum-ids

rekey-split-parquet-spectrum-ids: \
	rekey-pxd019483-split-parquet-spectrum-ids \
	rekey-sbrodae-split-parquet-spectrum-ids \
	rekey-celegans-split-parquet-spectrum-ids

rekey-pxd019483-split-parquet-spectrum-ids:
	uv run python scripts/rekey_split_parquet_spectrum_ids.py --split-dir PXD019483_split_parquet

rekey-sbrodae-split-parquet-spectrum-ids:
	uv run python scripts/rekey_split_parquet_spectrum_ids.py --split-dir sbrodae_split_parquet

rekey-celegans-split-parquet-spectrum-ids:
	uv run python scripts/rekey_split_parquet_spectrum_ids.py --split-dir celegans_split_parquet

# ═══════════════════════════════════════════════════════
# InstaNovo — data lives in winnow repo as parquet + CSV
# ═══════════════════════════════════════════════════════

# -- helaqc --
.PHONY: train-instanovo-helaqc plot-feature-investigation-instanovo-helaqc predict-instanovo-helaqc-test predict-instanovo-helaqc-unlabelled predict-instanovo-helaqc-raw_less_train plot-instanovo-helaqc

train-instanovo-helaqc:
	uv run winnow compute-features --config-dir configs/instanovo \
	dataset.spectrum_path_or_directory=helaqc_split_parquet/annotated_train.parquet \
	dataset.predictions_path=held_out_projects/biological_validation/annotated_predictions/dataset-helaqc-annotated-0000-0001.csv \
	training_matrix_output_path=$(ANALYSIS_MODELS)/instanovo_helaqc/features_train.parquet \
	metadata_output_path=$(ANALYSIS_MODELS)/instanovo_helaqc/metadata_train.parquet \
	$(ANALYSIS_KOINA_OVERRIDES)
	uv run winnow compute-features --config-dir configs/instanovo \
	dataset.spectrum_path_or_directory=helaqc_split_parquet/annotated_val.parquet \
	dataset.predictions_path=held_out_projects/biological_validation/annotated_predictions/dataset-helaqc-annotated-0000-0001.csv \
	training_matrix_output_path=$(ANALYSIS_MODELS)/instanovo_helaqc/features_val.parquet \
	metadata_output_path=$(ANALYSIS_MODELS)/instanovo_helaqc/metadata_val.parquet \
	$(ANALYSIS_KOINA_OVERRIDES)
	uv run winnow train --config-dir configs/instanovo \
	features_path=$(ANALYSIS_MODELS)/instanovo_helaqc/features_train.parquet \
	val_features_path=$(ANALYSIS_MODELS)/instanovo_helaqc/features_val.parquet \
	model_output_dir=$(ANALYSIS_MODELS)/instanovo_helaqc \
	dataset_output_path=null \
	irt_regressor_output_path=$(ANALYSIS_MODELS)/instanovo_helaqc/irt_regressors.safetensors \
	training_history_path=$(ANALYSIS_MODELS)/instanovo_helaqc/training_history.json

plot-feature-investigation-instanovo-helaqc:
	uv run python scripts/plot_feature_investigation.py \
		--features-train $(ANALYSIS_MODELS)/instanovo_helaqc/features_train.parquet \
		--features-val $(ANALYSIS_MODELS)/instanovo_helaqc/features_val.parquet \
		--output-dir $(ANALYSIS_MODELS)/instanovo_helaqc/feature_investigation_plots

predict-instanovo-helaqc-test:
	uv run winnow predict --config-dir configs/instanovo \
	dataset.spectrum_path_or_directory=helaqc_split_parquet/annotated_test.parquet \
	dataset.predictions_path=held_out_projects/biological_validation/annotated_predictions/dataset-helaqc-annotated-0000-0001.csv \
	calibrator.pretrained_model_name_or_path=$(ANALYSIS_MODELS)/instanovo_helaqc \
	output_folder=$(ANALYSIS_RESULTS)/instanovo_helaqc_predictions_test/ \
	$(ANALYSIS_PREDICT_OVERRIDES) $(ANALYSIS_KOINA_OVERRIDES)
	$(call upload_analysis_predictions,$(ANALYSIS_RESULTS)/instanovo_helaqc_predictions_test)

predict-instanovo-helaqc-unlabelled:
	uv run winnow predict --config-dir configs/instanovo \
	dataset.spectrum_path_or_directory=helaqc_split_parquet/raw_unlabelled.parquet \
	dataset.predictions_path=held_out_projects/biological_validation/raw_predictions/dataset-helaqc-raw-0000-0001.csv \
	calibrator.pretrained_model_name_or_path=$(ANALYSIS_MODELS)/instanovo_helaqc \
	output_folder=$(ANALYSIS_RESULTS)/instanovo_helaqc_predictions_unlabelled/ \
	$(ANALYSIS_PREDICT_OVERRIDES) $(ANALYSIS_KOINA_OVERRIDES)
	$(call upload_analysis_predictions,$(ANALYSIS_RESULTS)/instanovo_helaqc_predictions_unlabelled)

predict-instanovo-helaqc-raw_less_train:
	uv run winnow predict --config-dir configs/instanovo \
	dataset.spectrum_path_or_directory=helaqc_split_parquet/raw_less_train.parquet \
	dataset.predictions_path=held_out_projects/biological_validation/raw_predictions/dataset-helaqc-raw-0000-0001.csv \
	calibrator.pretrained_model_name_or_path=$(ANALYSIS_MODELS)/instanovo_helaqc \
	output_folder=$(ANALYSIS_RESULTS)/instanovo_helaqc_predictions_raw_less_train/ \
	$(ANALYSIS_PREDICT_OVERRIDES) $(ANALYSIS_KOINA_OVERRIDES)
	$(call upload_analysis_predictions,$(ANALYSIS_RESULTS)/instanovo_helaqc_predictions_raw_less_train)

plot-instanovo-helaqc:
	uv run python scripts/plot_analysis.py \
		--predictions-dir $(ANALYSIS_RESULTS)/instanovo_helaqc_predictions_test \
		--split test --label-mode labelled \
		--fasta $(ANALYSIS_FASTA_helaqc) \
		--model-dir $(ANALYSIS_MODELS)/instanovo_helaqc
	uv run python scripts/plot_analysis.py \
		--predictions-dir $(ANALYSIS_RESULTS)/instanovo_helaqc_predictions_unlabelled \
		--split unlabelled --label-mode unlabelled \
		--fasta $(ANALYSIS_FASTA_helaqc)
	uv run python scripts/plot_analysis.py \
		--predictions-dir $(ANALYSIS_RESULTS)/instanovo_helaqc_predictions_raw_less_train \
		--split raw_less_train --label-mode unlabelled \
		--fasta $(ANALYSIS_FASTA_helaqc)

# -- celegans (MGF-matched predictions in celegans_split_parquet/*.csv) --
.PHONY: train-instanovo-celegans predict-instanovo-celegans-test predict-instanovo-celegans-unlabelled predict-instanovo-celegans-raw_less_train plot-instanovo-celegans

train-instanovo-celegans:
	uv run winnow compute-features --config-dir configs/instanovo \
	dataset.spectrum_path_or_directory=celegans_split_parquet/annotated_train.parquet \
	dataset.predictions_path=held_out_projects/lcfm/PXD014877_predictions/PXD014877.csv \
	training_matrix_output_path=$(ANALYSIS_MODELS)/instanovo_celegans/features_train.parquet \
	+metadata_output_path=$(ANALYSIS_MODELS)/instanovo_celegans/metadata_train.parquet \
	$(ANALYSIS_KOINA_OVERRIDES)
	uv run winnow compute-features --config-dir configs/instanovo \
	dataset.spectrum_path_or_directory=celegans_split_parquet/annotated_val.parquet \
	dataset.predictions_path=held_out_projects/lcfm/PXD014877_predictions/PXD014877.csv \
	training_matrix_output_path=$(ANALYSIS_MODELS)/instanovo_celegans/features_val.parquet \
	+metadata_output_path=$(ANALYSIS_MODELS)/instanovo_celegans/metadata_val.parquet \
	$(ANALYSIS_KOINA_OVERRIDES)
	uv run winnow train --config-dir configs/instanovo \
	features_path=$(ANALYSIS_MODELS)/instanovo_celegans/features_train.parquet \
	val_features_path=$(ANALYSIS_MODELS)/instanovo_celegans/features_val.parquet \
	model_output_dir=$(ANALYSIS_MODELS)/instanovo_celegans \
	dataset_output_path=null \
	irt_regressor_output_path=$(ANALYSIS_MODELS)/instanovo_celegans/irt_regressors.safetensors \
	training_history_path=$(ANALYSIS_MODELS)/instanovo_celegans/training_history.json

predict-instanovo-celegans-test:
	uv run winnow predict --config-dir configs/instanovo \
	dataset.spectrum_path_or_directory=celegans_split_parquet/annotated_test.parquet \
	dataset.predictions_path=held_out_projects/lcfm/PXD014877_predictions/PXD014877.csv \
	calibrator.pretrained_model_name_or_path=$(ANALYSIS_MODELS)/instanovo_celegans \
	output_folder=$(ANALYSIS_RESULTS)/instanovo_celegans_predictions_test/ \
	$(ANALYSIS_PREDICT_OVERRIDES) $(ANALYSIS_KOINA_OVERRIDES)
	$(call upload_analysis_predictions,$(ANALYSIS_RESULTS)/instanovo_celegans_predictions_test)

predict-instanovo-celegans-unlabelled:
	uv run winnow predict --config-dir configs/instanovo \
	dataset.spectrum_path_or_directory=celegans_split_parquet/raw_unlabelled.parquet \
	dataset.predictions_path=held_out_projects/acfm/PXD014877_predictions/PXD014877.csv \
	calibrator.pretrained_model_name_or_path=$(ANALYSIS_MODELS)/instanovo_celegans \
	output_folder=$(ANALYSIS_RESULTS)/instanovo_celegans_predictions_unlabelled/ \
	$(ANALYSIS_PREDICT_OVERRIDES) $(ANALYSIS_KOINA_OVERRIDES)
	$(call upload_analysis_predictions,$(ANALYSIS_RESULTS)/instanovo_celegans_predictions_unlabelled)

predict-instanovo-celegans-raw_less_train:
	uv run winnow predict --config-dir configs/instanovo \
	dataset.spectrum_path_or_directory=celegans_split_parquet/raw_less_train.parquet \
	dataset.predictions_path=held_out_projects/acfm/PXD014877_predictions/PXD014877.csv \
	calibrator.pretrained_model_name_or_path=$(ANALYSIS_MODELS)/instanovo_celegans \
	output_folder=$(ANALYSIS_RESULTS)/instanovo_celegans_predictions_raw_less_train/ \
	$(ANALYSIS_PREDICT_OVERRIDES) $(ANALYSIS_KOINA_OVERRIDES)
	$(call upload_analysis_predictions,$(ANALYSIS_RESULTS)/instanovo_celegans_predictions_raw_less_train)

plot-instanovo-celegans:
	uv run python scripts/plot_analysis.py \
		--predictions-dir $(ANALYSIS_RESULTS)/instanovo_celegans_predictions_test \
		--split test --label-mode labelled \
		--fasta $(ANALYSIS_FASTA_celegans) \
		--model-dir $(ANALYSIS_MODELS)/instanovo_celegans
	uv run python scripts/plot_analysis.py \
		--predictions-dir $(ANALYSIS_RESULTS)/instanovo_celegans_predictions_unlabelled \
		--split unlabelled --label-mode unlabelled \
		--fasta $(ANALYSIS_FASTA_celegans)
	uv run python scripts/plot_analysis.py \
		--predictions-dir $(ANALYSIS_RESULTS)/instanovo_celegans_predictions_raw_less_train \
		--split raw_less_train --label-mode unlabelled \
		--fasta $(ANALYSIS_FASTA_celegans)

# -- PXD019483 (HepG2; MGF-matched predictions in PXD019483_split_parquet/*.csv) --
.PHONY: train-instanovo-pxd019483 predict-instanovo-pxd019483-test predict-instanovo-pxd019483-unlabelled plot-instanovo-pxd019483

train-instanovo-pxd019483: rekey-pxd019483-split-parquet-spectrum-ids
	uv run winnow compute-features --config-dir configs/instanovo \
	dataset.spectrum_path_or_directory=PXD019483_split_parquet/annotated_train.parquet \
	dataset.predictions_path=PXD019483_split_parquet/annotated_train.csv \
	training_matrix_output_path=$(ANALYSIS_MODELS)/instanovo_pxd019483/features_train.parquet \
	+metadata_output_path=$(ANALYSIS_MODELS)/instanovo_pxd019483/metadata_train.parquet \
	$(ANALYSIS_KOINA_OVERRIDES)
	uv run winnow compute-features --config-dir configs/instanovo \
	dataset.spectrum_path_or_directory=PXD019483_split_parquet/annotated_val.parquet \
	dataset.predictions_path=PXD019483_split_parquet/annotated_val.csv \
	training_matrix_output_path=$(ANALYSIS_MODELS)/instanovo_pxd019483/features_val.parquet \
	+metadata_output_path=$(ANALYSIS_MODELS)/instanovo_pxd019483/metadata_val.parquet \
	$(ANALYSIS_KOINA_OVERRIDES)
	uv run winnow train --config-dir configs/instanovo \
	features_path=$(ANALYSIS_MODELS)/instanovo_pxd019483/features_train.parquet \
	val_features_path=$(ANALYSIS_MODELS)/instanovo_pxd019483/features_val.parquet \
	model_output_dir=$(ANALYSIS_MODELS)/instanovo_pxd019483 \
	dataset_output_path=null \
	irt_regressor_output_path=$(ANALYSIS_MODELS)/instanovo_pxd019483/irt_regressors.safetensors \
	training_history_path=$(ANALYSIS_MODELS)/instanovo_pxd019483/training_history.json

predict-instanovo-pxd019483-test: rekey-pxd019483-split-parquet-spectrum-ids
	uv run winnow predict --config-dir configs/instanovo \
	dataset.spectrum_path_or_directory=PXD019483_split_parquet/annotated_test.parquet \
	dataset.predictions_path=PXD019483_split_parquet/annotated_test.csv \
	calibrator.pretrained_model_name_or_path=$(ANALYSIS_MODELS)/instanovo_pxd019483 \
	output_folder=$(ANALYSIS_RESULTS)/instanovo_pxd019483_predictions_test/ \
	$(ANALYSIS_PREDICT_OVERRIDES) $(ANALYSIS_KOINA_OVERRIDES)

predict-instanovo-pxd019483-unlabelled: rekey-pxd019483-split-parquet-spectrum-ids
	uv run winnow predict --config-dir configs/instanovo \
	dataset.spectrum_path_or_directory=PXD019483_split_parquet/raw_unlabelled.parquet \
	dataset.predictions_path=PXD019483_split_parquet/raw_unlabelled.csv \
	calibrator.pretrained_model_name_or_path=$(ANALYSIS_MODELS)/instanovo_pxd019483 \
	output_folder=$(ANALYSIS_RESULTS)/instanovo_pxd019483_predictions_unlabelled/ \
	$(ANALYSIS_PREDICT_OVERRIDES) $(ANALYSIS_KOINA_OVERRIDES)

plot-instanovo-pxd019483:
	uv run python scripts/plot_analysis.py \
		--predictions-dir $(ANALYSIS_RESULTS)/instanovo_pxd019483_predictions_test \
		--split test --label-mode labelled \
		--fasta $(ANALYSIS_FASTA_pxd019483) \
		--model-dir $(ANALYSIS_MODELS)/instanovo_pxd019483
	uv run python scripts/plot_analysis.py \
		--predictions-dir $(ANALYSIS_RESULTS)/instanovo_pxd019483_predictions_unlabelled \
		--split unlabelled --label-mode unlabelled \
		--fasta $(ANALYSIS_FASTA_pxd019483)

# -- sbrodae (MGF-matched predictions in sbrodae_split_parquet/*.csv) --
.PHONY: train-instanovo-sbrodae predict-instanovo-sbrodae-test predict-instanovo-sbrodae-unlabelled plot-instanovo-sbrodae

train-instanovo-sbrodae: rekey-sbrodae-split-parquet-spectrum-ids
	uv run winnow compute-features --config-dir configs/instanovo \
	dataset.spectrum_path_or_directory=sbrodae_split_parquet/annotated_train.parquet \
	dataset.predictions_path=sbrodae_split_parquet/annotated_train.csv \
	training_matrix_output_path=$(ANALYSIS_MODELS)/instanovo_sbrodae/features_train.parquet \
	+metadata_output_path=$(ANALYSIS_MODELS)/instanovo_sbrodae/metadata_train.parquet \
	$(ANALYSIS_KOINA_OVERRIDES)
	uv run winnow compute-features --config-dir configs/instanovo \
	dataset.spectrum_path_or_directory=sbrodae_split_parquet/annotated_val.parquet \
	dataset.predictions_path=sbrodae_split_parquet/annotated_val.csv \
	training_matrix_output_path=$(ANALYSIS_MODELS)/instanovo_sbrodae/features_val.parquet \
	+metadata_output_path=$(ANALYSIS_MODELS)/instanovo_sbrodae/metadata_val.parquet \
	$(ANALYSIS_KOINA_OVERRIDES)
	uv run winnow train --config-dir configs/instanovo \
	features_path=$(ANALYSIS_MODELS)/instanovo_sbrodae/features_train.parquet \
	val_features_path=$(ANALYSIS_MODELS)/instanovo_sbrodae/features_val.parquet \
	model_output_dir=$(ANALYSIS_MODELS)/instanovo_sbrodae \
	dataset_output_path=null \
	irt_regressor_output_path=$(ANALYSIS_MODELS)/instanovo_sbrodae/irt_regressors.safetensors \
	training_history_path=$(ANALYSIS_MODELS)/instanovo_sbrodae/training_history.json

predict-instanovo-sbrodae-test: rekey-sbrodae-split-parquet-spectrum-ids
	uv run winnow predict --config-dir configs/instanovo \
	dataset.spectrum_path_or_directory=sbrodae_split_parquet/annotated_test.parquet \
	dataset.predictions_path=sbrodae_split_parquet/annotated_test.csv \
	calibrator.pretrained_model_name_or_path=$(ANALYSIS_MODELS)/instanovo_sbrodae \
	output_folder=$(ANALYSIS_RESULTS)/instanovo_sbrodae_predictions_test/ \
	$(ANALYSIS_PREDICT_OVERRIDES) $(ANALYSIS_KOINA_OVERRIDES)

predict-instanovo-sbrodae-unlabelled: rekey-sbrodae-split-parquet-spectrum-ids
	uv run winnow predict --config-dir configs/instanovo \
	dataset.spectrum_path_or_directory=sbrodae_split_parquet/raw_unlabelled.parquet \
	dataset.predictions_path=sbrodae_split_parquet/raw_unlabelled.csv \
	calibrator.pretrained_model_name_or_path=$(ANALYSIS_MODELS)/instanovo_sbrodae \
	output_folder=$(ANALYSIS_RESULTS)/instanovo_sbrodae_predictions_unlabelled/ \
	$(ANALYSIS_PREDICT_OVERRIDES) $(ANALYSIS_KOINA_OVERRIDES)

plot-instanovo-sbrodae:
	uv run python scripts/plot_analysis.py \
		--predictions-dir $(ANALYSIS_RESULTS)/instanovo_sbrodae_predictions_test \
		--split test --label-mode labelled \
		--fasta $(ANALYSIS_FASTA_sbrodae) \
		--model-dir $(ANALYSIS_MODELS)/instanovo_sbrodae
	uv run python scripts/plot_analysis.py \
		--predictions-dir $(ANALYSIS_RESULTS)/instanovo_sbrodae_predictions_unlabelled \
		--split unlabelled --label-mode unlabelled \
		--fasta $(ANALYSIS_FASTA_sbrodae)

# ═══════════════════════════════════════════════════════
# Casanovo — data lives in external repo as MGF + mztab
# ═══════════════════════════════════════════════════════

# -- helaqc --
.PHONY: train-casanovo-helaqc predict-casanovo-helaqc-test predict-casanovo-helaqc-unlabelled predict-casanovo-helaqc-raw_less_train plot-casanovo-helaqc

train-casanovo-helaqc:
	uv run winnow compute-features --config-dir configs/casanovo \
	dataset.spectrum_path_or_directory=$(CASANOVO_ROOT)/helaqc_mgf/helaqc_annotated_train.mgf \
	dataset.predictions_path=$(CASANOVO_ROOT)/predictions/helaqc/helaqc_annotated_train.mztab \
	training_matrix_output_path=$(ANALYSIS_MODELS)/casanovo_helaqc/features_train.parquet \
	$(ANALYSIS_KOINA_OVERRIDES)
	uv run winnow compute-features --config-dir configs/casanovo \
	dataset.spectrum_path_or_directory=$(CASANOVO_ROOT)/helaqc_mgf/helaqc_annotated_val.mgf \
	dataset.predictions_path=$(CASANOVO_ROOT)/predictions/helaqc/helaqc_annotated_val.mztab \
	training_matrix_output_path=$(ANALYSIS_MODELS)/casanovo_helaqc/features_val.parquet \
	$(ANALYSIS_KOINA_OVERRIDES)
	uv run winnow train --config-dir configs/casanovo \
	features_path=$(ANALYSIS_MODELS)/casanovo_helaqc/features_train.parquet \
	val_features_path=$(ANALYSIS_MODELS)/casanovo_helaqc/features_val.parquet \
	model_output_dir=$(ANALYSIS_MODELS)/casanovo_helaqc \
	dataset_output_path=null \
	irt_regressor_output_path=$(ANALYSIS_MODELS)/casanovo_helaqc/irt_regressors.safetensors \
	training_history_path=$(ANALYSIS_MODELS)/casanovo_helaqc/training_history.json

predict-casanovo-helaqc-test:
	uv run winnow predict --config-dir configs/casanovo \
	dataset.spectrum_path_or_directory=$(CASANOVO_ROOT)/helaqc_mgf/helaqc_annotated_test.mgf \
	dataset.predictions_path=$(CASANOVO_ROOT)/predictions/helaqc/helaqc_annotated_test.mztab \
	calibrator.pretrained_model_name_or_path=$(ANALYSIS_MODELS)/casanovo_helaqc \
	output_folder=$(ANALYSIS_RESULTS)/casanovo_helaqc_predictions_test/ \
	$(ANALYSIS_PREDICT_OVERRIDES) $(ANALYSIS_KOINA_OVERRIDES)
	$(call upload_analysis_predictions,$(ANALYSIS_RESULTS)/casanovo_helaqc_predictions_test)

predict-casanovo-helaqc-unlabelled:
	uv run winnow predict --config-dir configs/casanovo \
	dataset.spectrum_path_or_directory=$(CASANOVO_ROOT)/helaqc_mgf/helaqc_raw_unlabelled.mgf \
	dataset.predictions_path=$(CASANOVO_ROOT)/predictions/helaqc/helaqc_raw_unlabelled.mztab \
	calibrator.pretrained_model_name_or_path=$(ANALYSIS_MODELS)/casanovo_helaqc \
	output_folder=$(ANALYSIS_RESULTS)/casanovo_helaqc_predictions_unlabelled/ \
	$(ANALYSIS_PREDICT_OVERRIDES) $(ANALYSIS_KOINA_OVERRIDES)
	$(call upload_analysis_predictions,$(ANALYSIS_RESULTS)/casanovo_helaqc_predictions_unlabelled)

predict-casanovo-helaqc-raw_less_train:
	uv run winnow predict --config-dir configs/casanovo \
	dataset.spectrum_path_or_directory=$(CASANOVO_ROOT)/helaqc_mgf/helaqc_raw_less_train.mgf \
	dataset.predictions_path=$(CASANOVO_ROOT)/predictions/helaqc/helaqc_raw_less_train.mztab \
	calibrator.pretrained_model_name_or_path=$(ANALYSIS_MODELS)/casanovo_helaqc \
	output_folder=$(ANALYSIS_RESULTS)/casanovo_helaqc_predictions_raw_less_train/ \
	$(ANALYSIS_PREDICT_OVERRIDES) $(ANALYSIS_KOINA_OVERRIDES)
	$(call upload_analysis_predictions,$(ANALYSIS_RESULTS)/casanovo_helaqc_predictions_raw_less_train)

plot-casanovo-helaqc:
	uv run python scripts/plot_analysis.py \
		--predictions-dir $(ANALYSIS_RESULTS)/casanovo_helaqc_predictions_test \
		--split test --label-mode labelled \
		--fasta $(ANALYSIS_FASTA_helaqc) \
		--model-dir $(ANALYSIS_MODELS)/casanovo_helaqc
	uv run python scripts/plot_analysis.py \
		--predictions-dir $(ANALYSIS_RESULTS)/casanovo_helaqc_predictions_unlabelled \
		--split unlabelled --label-mode unlabelled \
		--fasta $(ANALYSIS_FASTA_helaqc)
	uv run python scripts/plot_analysis.py \
		--predictions-dir $(ANALYSIS_RESULTS)/casanovo_helaqc_predictions_raw_less_train \
		--split raw_less_train --label-mode unlabelled \
		--fasta $(ANALYSIS_FASTA_helaqc)

# -- celegans --
.PHONY: train-casanovo-celegans predict-casanovo-celegans-test predict-casanovo-celegans-unlabelled predict-casanovo-celegans-raw_less_train plot-casanovo-celegans

train-casanovo-celegans:
	uv run winnow compute-features --config-dir configs/casanovo \
	dataset.spectrum_path_or_directory=$(CASANOVO_ROOT)/celegans_mgf/celegans_annotated_train.mgf \
	dataset.predictions_path=$(CASANOVO_ROOT)/predictions/celegans/celegans_annotated_train.mztab \
	training_matrix_output_path=$(ANALYSIS_MODELS)/casanovo_celegans/features_train.parquet \
	$(ANALYSIS_KOINA_OVERRIDES)
	uv run winnow compute-features --config-dir configs/casanovo \
	dataset.spectrum_path_or_directory=$(CASANOVO_ROOT)/celegans_mgf/celegans_annotated_val.mgf \
	dataset.predictions_path=$(CASANOVO_ROOT)/predictions/celegans/celegans_annotated_val.mztab \
	training_matrix_output_path=$(ANALYSIS_MODELS)/casanovo_celegans/features_val.parquet \
	$(ANALYSIS_KOINA_OVERRIDES)
	uv run winnow train --config-dir configs/casanovo \
	features_path=$(ANALYSIS_MODELS)/casanovo_celegans/features_train.parquet \
	val_features_path=$(ANALYSIS_MODELS)/casanovo_celegans/features_val.parquet \
	model_output_dir=$(ANALYSIS_MODELS)/casanovo_celegans \
	dataset_output_path=null \
	irt_regressor_output_path=$(ANALYSIS_MODELS)/casanovo_celegans/irt_regressors.safetensors \
	training_history_path=$(ANALYSIS_MODELS)/casanovo_celegans/training_history.json

predict-casanovo-celegans-test:
	uv run winnow predict --config-dir configs/casanovo \
	dataset.spectrum_path_or_directory=$(CASANOVO_ROOT)/celegans_mgf/celegans_annotated_test.mgf \
	dataset.predictions_path=$(CASANOVO_ROOT)/predictions/celegans/celegans_annotated_test.mztab \
	calibrator.pretrained_model_name_or_path=$(ANALYSIS_MODELS)/casanovo_celegans \
	output_folder=$(ANALYSIS_RESULTS)/casanovo_celegans_predictions_test/ \
	$(ANALYSIS_PREDICT_OVERRIDES) $(ANALYSIS_KOINA_OVERRIDES)
	$(call upload_analysis_predictions,$(ANALYSIS_RESULTS)/casanovo_celegans_predictions_test)

predict-casanovo-celegans-unlabelled:
	uv run winnow predict --config-dir configs/casanovo \
	dataset.spectrum_path_or_directory=$(CASANOVO_ROOT)/celegans_mgf/celegans_raw_unlabelled.mgf \
	dataset.predictions_path=$(CASANOVO_ROOT)/predictions/celegans/celegans_raw_unlabelled.mztab \
	calibrator.pretrained_model_name_or_path=$(ANALYSIS_MODELS)/casanovo_celegans \
	output_folder=$(ANALYSIS_RESULTS)/casanovo_celegans_predictions_unlabelled/ \
	$(ANALYSIS_PREDICT_OVERRIDES) $(ANALYSIS_KOINA_OVERRIDES)
	$(call upload_analysis_predictions,$(ANALYSIS_RESULTS)/casanovo_celegans_predictions_unlabelled)

predict-casanovo-celegans-raw_less_train:
	uv run winnow predict --config-dir configs/casanovo \
	dataset.spectrum_path_or_directory=$(CASANOVO_ROOT)/celegans_mgf/celegans_raw_less_train.mgf \
	dataset.predictions_path=$(CASANOVO_ROOT)/predictions/celegans/celegans_raw_less_train.mztab \
	calibrator.pretrained_model_name_or_path=$(ANALYSIS_MODELS)/casanovo_celegans \
	output_folder=$(ANALYSIS_RESULTS)/casanovo_celegans_predictions_raw_less_train/ \
	$(ANALYSIS_PREDICT_OVERRIDES) $(ANALYSIS_KOINA_OVERRIDES)
	$(call upload_analysis_predictions,$(ANALYSIS_RESULTS)/casanovo_celegans_predictions_raw_less_train)

plot-casanovo-celegans:
	uv run python scripts/plot_analysis.py \
		--predictions-dir $(ANALYSIS_RESULTS)/casanovo_celegans_predictions_test \
		--split test --label-mode labelled \
		--fasta $(ANALYSIS_FASTA_celegans) \
		--model-dir $(ANALYSIS_MODELS)/casanovo_celegans
	uv run python scripts/plot_analysis.py \
		--predictions-dir $(ANALYSIS_RESULTS)/casanovo_celegans_predictions_unlabelled \
		--split unlabelled --label-mode unlabelled \
		--fasta $(ANALYSIS_FASTA_celegans)
	uv run python scripts/plot_analysis.py \
		--predictions-dir $(ANALYSIS_RESULTS)/casanovo_celegans_predictions_raw_less_train \
		--split raw_less_train --label-mode unlabelled \
		--fasta $(ANALYSIS_FASTA_celegans)

# ═══════════════════════════════════════════════════════
# PrimeNovo — data lives in external repo as MGF + TSV
# ═══════════════════════════════════════════════════════

# -- helaqc --
.PHONY: train-primenovo-helaqc predict-primenovo-helaqc-test predict-primenovo-helaqc-unlabelled predict-primenovo-helaqc-raw_less_train plot-primenovo-helaqc

train-primenovo-helaqc:
	uv run winnow compute-features --config-dir configs/primenovo \
	dataset.spectrum_path_or_directory=$(PRIMENOVO_ROOT)/helaqc_mgf/helaqc_annotated_train.mgf \
	dataset.predictions_path=$(PRIMENOVO_ROOT)/predictions/helaqc/helaqc_denovo_annotated_train.tsv \
	training_matrix_output_path=$(ANALYSIS_MODELS)/primenovo_helaqc/features_train.parquet \
	$(ANALYSIS_KOINA_OVERRIDES)
	uv run winnow compute-features --config-dir configs/primenovo \
	dataset.spectrum_path_or_directory=$(PRIMENOVO_ROOT)/helaqc_mgf/helaqc_annotated_val.mgf \
	dataset.predictions_path=$(PRIMENOVO_ROOT)/predictions/helaqc/helaqc_denovo_annotated_val.tsv \
	training_matrix_output_path=$(ANALYSIS_MODELS)/primenovo_helaqc/features_val.parquet \
	$(ANALYSIS_KOINA_OVERRIDES)
	uv run winnow train --config-dir configs/primenovo \
	features_path=$(ANALYSIS_MODELS)/primenovo_helaqc/features_train.parquet \
	val_features_path=$(ANALYSIS_MODELS)/primenovo_helaqc/features_val.parquet \
	model_output_dir=$(ANALYSIS_MODELS)/primenovo_helaqc \
	dataset_output_path=null \
	irt_regressor_output_path=$(ANALYSIS_MODELS)/primenovo_helaqc/irt_regressors.safetensors \
	training_history_path=$(ANALYSIS_MODELS)/primenovo_helaqc/training_history.json

predict-primenovo-helaqc-test:
	uv run winnow predict --config-dir configs/primenovo \
	dataset.spectrum_path_or_directory=$(PRIMENOVO_ROOT)/helaqc_mgf/helaqc_annotated_test.mgf \
	dataset.predictions_path=$(PRIMENOVO_ROOT)/predictions/helaqc/helaqc_denovo_annotated_test.tsv \
	calibrator.pretrained_model_name_or_path=$(ANALYSIS_MODELS)/primenovo_helaqc \
	output_folder=$(ANALYSIS_RESULTS)/primenovo_helaqc_predictions_test/ \
	$(ANALYSIS_PREDICT_OVERRIDES) $(ANALYSIS_KOINA_OVERRIDES)
	$(call upload_analysis_predictions,$(ANALYSIS_RESULTS)/primenovo_helaqc_predictions_test)

predict-primenovo-helaqc-unlabelled:
	uv run winnow predict --config-dir configs/primenovo \
	dataset.spectrum_path_or_directory=$(PRIMENOVO_ROOT)/helaqc_mgf/helaqc_raw_unlabelled.mgf \
	dataset.predictions_path=$(PRIMENOVO_ROOT)/predictions/helaqc/helaqc_denovo_raw_unlabelled.tsv \
	calibrator.pretrained_model_name_or_path=$(ANALYSIS_MODELS)/primenovo_helaqc \
	output_folder=$(ANALYSIS_RESULTS)/primenovo_helaqc_predictions_unlabelled/ \
	$(ANALYSIS_PREDICT_OVERRIDES) $(ANALYSIS_KOINA_OVERRIDES)
	$(call upload_analysis_predictions,$(ANALYSIS_RESULTS)/primenovo_helaqc_predictions_unlabelled)

predict-primenovo-helaqc-raw_less_train:
	uv run winnow predict --config-dir configs/primenovo \
	dataset.spectrum_path_or_directory=$(PRIMENOVO_ROOT)/helaqc_mgf/helaqc_raw_less_train.mgf \
	dataset.predictions_path=$(PRIMENOVO_ROOT)/predictions/helaqc/helaqc_denovo_raw_less_train.tsv \
	calibrator.pretrained_model_name_or_path=$(ANALYSIS_MODELS)/primenovo_helaqc \
	output_folder=$(ANALYSIS_RESULTS)/primenovo_helaqc_predictions_raw_less_train/ \
	$(ANALYSIS_PREDICT_OVERRIDES) $(ANALYSIS_KOINA_OVERRIDES)
	$(call upload_analysis_predictions,$(ANALYSIS_RESULTS)/primenovo_helaqc_predictions_raw_less_train)

plot-primenovo-helaqc:
	uv run python scripts/plot_analysis.py \
		--predictions-dir $(ANALYSIS_RESULTS)/primenovo_helaqc_predictions_test \
		--split test --label-mode labelled \
		--fasta $(ANALYSIS_FASTA_helaqc) \
		--model-dir $(ANALYSIS_MODELS)/primenovo_helaqc
	uv run python scripts/plot_analysis.py \
		--predictions-dir $(ANALYSIS_RESULTS)/primenovo_helaqc_predictions_unlabelled \
		--split unlabelled --label-mode unlabelled \
		--fasta $(ANALYSIS_FASTA_helaqc)
	uv run python scripts/plot_analysis.py \
		--predictions-dir $(ANALYSIS_RESULTS)/primenovo_helaqc_predictions_raw_less_train \
		--split raw_less_train --label-mode unlabelled \
		--fasta $(ANALYSIS_FASTA_helaqc)

# -- celegans --
.PHONY: train-primenovo-celegans predict-primenovo-celegans-test predict-primenovo-celegans-unlabelled predict-primenovo-celegans-raw_less_train plot-primenovo-celegans

train-primenovo-celegans:
	uv run winnow compute-features --config-dir configs/primenovo \
	dataset.spectrum_path_or_directory=$(PRIMENOVO_ROOT)/celegans_mgf/celegans_annotated_train.mgf \
	dataset.predictions_path=$(PRIMENOVO_ROOT)/predictions/celegans/celegans_denovo_annotated_train.tsv \
	training_matrix_output_path=$(ANALYSIS_MODELS)/primenovo_celegans/features_train.parquet \
	$(ANALYSIS_KOINA_OVERRIDES)
	uv run winnow compute-features --config-dir configs/primenovo \
	dataset.spectrum_path_or_directory=$(PRIMENOVO_ROOT)/celegans_mgf/celegans_annotated_val.mgf \
	dataset.predictions_path=$(PRIMENOVO_ROOT)/predictions/celegans/celegans_denovo_annotated_val.tsv \
	training_matrix_output_path=$(ANALYSIS_MODELS)/primenovo_celegans/features_val.parquet \
	$(ANALYSIS_KOINA_OVERRIDES)
	uv run winnow train --config-dir configs/primenovo \
	features_path=$(ANALYSIS_MODELS)/primenovo_celegans/features_train.parquet \
	val_features_path=$(ANALYSIS_MODELS)/primenovo_celegans/features_val.parquet \
	model_output_dir=$(ANALYSIS_MODELS)/primenovo_celegans \
	dataset_output_path=null \
	irt_regressor_output_path=$(ANALYSIS_MODELS)/primenovo_celegans/irt_regressors.safetensors \
	training_history_path=$(ANALYSIS_MODELS)/primenovo_celegans/training_history.json

predict-primenovo-celegans-test:
	uv run winnow predict --config-dir configs/primenovo \
	dataset.spectrum_path_or_directory=$(PRIMENOVO_ROOT)/celegans_mgf/celegans_annotated_test.mgf \
	dataset.predictions_path=$(PRIMENOVO_ROOT)/predictions/celegans/celegans_denovo_annotated_test.tsv \
	calibrator.pretrained_model_name_or_path=$(ANALYSIS_MODELS)/primenovo_celegans \
	output_folder=$(ANALYSIS_RESULTS)/primenovo_celegans_predictions_test/ \
	$(ANALYSIS_PREDICT_OVERRIDES) $(ANALYSIS_KOINA_OVERRIDES)
	$(call upload_analysis_predictions,$(ANALYSIS_RESULTS)/primenovo_celegans_predictions_test)

predict-primenovo-celegans-unlabelled:
	uv run winnow predict --config-dir configs/primenovo \
	dataset.spectrum_path_or_directory=$(PRIMENOVO_ROOT)/celegans_mgf/celegans_raw_unlabelled.mgf \
	dataset.predictions_path=$(PRIMENOVO_ROOT)/predictions/celegans/celegans_denovo_raw_unlabelled.tsv \
	calibrator.pretrained_model_name_or_path=$(ANALYSIS_MODELS)/primenovo_celegans \
	output_folder=$(ANALYSIS_RESULTS)/primenovo_celegans_predictions_unlabelled/ \
	$(ANALYSIS_PREDICT_OVERRIDES) $(ANALYSIS_KOINA_OVERRIDES)
	$(call upload_analysis_predictions,$(ANALYSIS_RESULTS)/primenovo_celegans_predictions_unlabelled)

predict-primenovo-celegans-raw_less_train:
	uv run winnow predict --config-dir configs/primenovo \
	dataset.spectrum_path_or_directory=$(PRIMENOVO_ROOT)/celegans_mgf/celegans_raw_less_train.mgf \
	dataset.predictions_path=$(PRIMENOVO_ROOT)/predictions/celegans/celegans_denovo_raw_less_train.tsv \
	calibrator.pretrained_model_name_or_path=$(ANALYSIS_MODELS)/primenovo_celegans \
	output_folder=$(ANALYSIS_RESULTS)/primenovo_celegans_predictions_raw_less_train/ \
	$(ANALYSIS_PREDICT_OVERRIDES) $(ANALYSIS_KOINA_OVERRIDES)
	$(call upload_analysis_predictions,$(ANALYSIS_RESULTS)/primenovo_celegans_predictions_raw_less_train)

plot-primenovo-celegans:
	uv run python scripts/plot_analysis.py \
		--predictions-dir $(ANALYSIS_RESULTS)/primenovo_celegans_predictions_test \
		--split test --label-mode labelled \
		--fasta $(ANALYSIS_FASTA_celegans) \
		--model-dir $(ANALYSIS_MODELS)/primenovo_celegans
	uv run python scripts/plot_analysis.py \
		--predictions-dir $(ANALYSIS_RESULTS)/primenovo_celegans_predictions_unlabelled \
		--split unlabelled --label-mode unlabelled \
		--fasta $(ANALYSIS_FASTA_celegans)
	uv run python scripts/plot_analysis.py \
		--predictions-dir $(ANALYSIS_RESULTS)/primenovo_celegans_predictions_raw_less_train \
		--split raw_less_train --label-mode unlabelled \
		--fasta $(ANALYSIS_FASTA_celegans)

# ═══════════════════════════════════════════════════════
# raw_less_train creation
# ═══════════════════════════════════════════════════════
.PHONY: create-raw-less-train create-raw-less-train-parquet create-raw-less-train-casanovo create-raw-less-train-primenovo

create-raw-less-train-parquet:
	uv run python scripts/create_raw_less_train.py parquet \
		--test-parquet helaqc_split_parquet/annotated_test.parquet \
		--raw-parquet helaqc_split_parquet/raw_unlabelled.parquet \
		--predictions-csv held_out_projects/biological_validation/raw_predictions/dataset-helaqc-raw-0000-0001.csv \
		--output-parquet helaqc_split_parquet/raw_less_train.parquet \
		--output-csv helaqc_split_parquet/raw_less_train.csv
	uv run python scripts/create_raw_less_train.py parquet \
		--test-parquet celegans_split_parquet/annotated_test.parquet \
		--raw-parquet celegans_split_parquet/raw_unlabelled.parquet \
		--predictions-csv held_out_projects/acfm/PXD014877_predictions/PXD014877.csv \
		--output-parquet celegans_split_parquet/raw_less_train.parquet \
		--output-csv celegans_split_parquet/raw_less_train.csv

create-raw-less-train-casanovo:
	python3 scripts/create_raw_less_train.py mgf-mztab \
		--test-mgf $(CASANOVO_ROOT)/helaqc_mgf/helaqc_annotated_test.mgf \
		--raw-mgf $(CASANOVO_ROOT)/helaqc_mgf/helaqc_raw_unlabelled.mgf \
		--test-mztab $(CASANOVO_ROOT)/predictions/helaqc/helaqc_annotated_test.mztab \
		--raw-mztab $(CASANOVO_ROOT)/predictions/helaqc/helaqc_raw_unlabelled.mztab \
		--output-mgf $(CASANOVO_ROOT)/helaqc_mgf/helaqc_raw_less_train.mgf \
		--output-mztab $(CASANOVO_ROOT)/predictions/helaqc/helaqc_raw_less_train.mztab
	python3 scripts/create_raw_less_train.py mgf-mztab \
		--test-mgf $(CASANOVO_ROOT)/celegans_mgf/celegans_annotated_test.mgf \
		--raw-mgf $(CASANOVO_ROOT)/celegans_mgf/celegans_raw_unlabelled.mgf \
		--test-mztab $(CASANOVO_ROOT)/predictions/celegans/celegans_annotated_test.mztab \
		--raw-mztab $(CASANOVO_ROOT)/predictions/celegans/celegans_raw_unlabelled.mztab \
		--output-mgf $(CASANOVO_ROOT)/celegans_mgf/celegans_raw_less_train.mgf \
		--output-mztab $(CASANOVO_ROOT)/predictions/celegans/celegans_raw_less_train.mztab

create-raw-less-train-primenovo:
	python3 scripts/create_raw_less_train.py mgf-tsv \
		--test-mgf $(PRIMENOVO_ROOT)/helaqc_mgf/helaqc_annotated_test.mgf \
		--raw-mgf $(PRIMENOVO_ROOT)/helaqc_mgf/helaqc_raw_unlabelled.mgf \
		--test-tsv $(PRIMENOVO_ROOT)/predictions/helaqc/helaqc_denovo_annotated_test.tsv \
		--raw-tsv $(PRIMENOVO_ROOT)/predictions/helaqc/helaqc_denovo_raw_unlabelled.tsv \
		--output-mgf $(PRIMENOVO_ROOT)/helaqc_mgf/helaqc_raw_less_train.mgf \
		--output-tsv $(PRIMENOVO_ROOT)/predictions/helaqc/helaqc_denovo_raw_less_train.tsv
	python3 scripts/create_raw_less_train.py mgf-tsv \
		--test-mgf $(PRIMENOVO_ROOT)/celegans_mgf/celegans_annotated_test.mgf \
		--raw-mgf $(PRIMENOVO_ROOT)/celegans_mgf/celegans_raw_unlabelled.mgf \
		--test-tsv $(PRIMENOVO_ROOT)/predictions/celegans/celegans_denovo_annotated_test.tsv \
		--raw-tsv $(PRIMENOVO_ROOT)/predictions/celegans/celegans_denovo_raw_unlabelled.tsv \
		--output-mgf $(PRIMENOVO_ROOT)/celegans_mgf/celegans_raw_less_train.mgf \
		--output-tsv $(PRIMENOVO_ROOT)/predictions/celegans/celegans_denovo_raw_less_train.tsv

create-raw-less-train: create-raw-less-train-parquet create-raw-less-train-casanovo create-raw-less-train-primenovo

# ═══════════════════════════════════════════════════════
# Upload / download analysis data for cluster runs
# ═══════════════════════════════════════════════════════

ANALYSIS_S3 ?= $(S3_BASE)/analysis

.PHONY: local-upload-analysis-data local-upload-analysis-models download-analysis-data download-analysis-models \
	upload-analysis-predictions-dir upload-analysis-plots upload-analysis-results download-analysis-results \
	analysis-pipeline

## Upload all datasets needed by train-all / predict-all / plot-all to S3.
## Run this locally before launching a cluster job.
local-upload-analysis-data:
	@echo "=== Uploading winnow-repo datasets ==="
	$(S3_CP) --recursive helaqc_split_parquet/          $(ANALYSIS_S3)/helaqc_split_parquet/ --profile winnow
	$(S3_CP) --recursive celegans_split_parquet/         $(ANALYSIS_S3)/celegans_split_parquet/ --profile winnow
	@echo "=== Uploading Casanovo data ==="
	$(S3_CP) --recursive $(CASANOVO_ROOT)/helaqc_mgf/    $(ANALYSIS_S3)/casanovo/helaqc_mgf/ --profile winnow
	$(S3_CP) --recursive $(CASANOVO_ROOT)/celegans_mgf/  $(ANALYSIS_S3)/casanovo/celegans_mgf/ --profile winnow
	$(S3_CP) --recursive $(CASANOVO_ROOT)/predictions/   $(ANALYSIS_S3)/casanovo/predictions/ --profile winnow
	@echo "=== Uploading PrimeNovo data ==="
	$(S3_CP) --recursive $(PRIMENOVO_ROOT)/helaqc_mgf/   $(ANALYSIS_S3)/primenovo/helaqc_mgf/ --profile winnow
	$(S3_CP) --recursive $(PRIMENOVO_ROOT)/celegans_mgf/ $(ANALYSIS_S3)/primenovo/celegans_mgf/ --profile winnow
	$(S3_CP) --recursive $(PRIMENOVO_ROOT)/predictions/  $(ANALYSIS_S3)/primenovo/predictions/ --profile winnow
	@echo "=== upload-analysis-data complete ==="

## Upload locally-trained models to S3 so the cluster can skip training.
local-upload-analysis-models:
	@echo "=== Uploading trained models ==="
	for m in instanovo_helaqc instanovo_celegans casanovo_helaqc casanovo_celegans primenovo_helaqc primenovo_celegans; do \
		if [ -d $(ANALYSIS_MODELS)/$$m ]; then \
			$(S3_CP) --recursive $(ANALYSIS_MODELS)/$$m/ $(ANALYSIS_S3)/models/$$m/ --profile winnow; \
		else \
			echo "WARN: $(ANALYSIS_MODELS)/$$m not found, skipping"; \
		fi; \
	done
	@echo "=== upload-analysis-models complete ==="

## Download pre-trained models from S3 (used on cluster to skip training).
download-analysis-models:
	@echo "=== Downloading trained models ==="
	for m in instanovo_helaqc instanovo_celegans casanovo_helaqc casanovo_celegans primenovo_helaqc primenovo_celegans; do \
		mkdir -p $(ANALYSIS_MODELS)/$$m; \
		$(S3_CP) --recursive $(ANALYSIS_S3)/models/$$m/ $(ANALYSIS_MODELS)/$$m/; \
	done
	@echo "=== download-analysis-models complete ==="

## Download analysis datasets from S3 into the container working directory.
## Sets CASANOVO_ROOT / PRIMENOVO_ROOT to analysis_data/{casanovo,primenovo}.
download-analysis-data:
	@echo "=== Downloading winnow-repo datasets ==="
	mkdir -p helaqc_split_parquet celegans_split_parquet fasta \
	         held_out_projects/biological_validation/annotated_predictions \
	         held_out_projects/biological_validation/raw_predictions \
	         held_out_projects/lcfm/PXD014877_predictions \
	         held_out_projects/acfm/PXD014877_predictions
	$(S3_CP) --recursive $(ANALYSIS_S3)/helaqc_split_parquet/          helaqc_split_parquet/
	$(S3_CP) --recursive $(ANALYSIS_S3)/celegans_split_parquet/         celegans_split_parquet/
	@# Predictions + FASTA live under held_out_projects/ on S3 (not under analysis/).
	$(S3_CP) --recursive $(HELD_OUT_S3)/biological_validation/annotated_predictions/ \
	         held_out_projects/biological_validation/annotated_predictions/
	$(S3_CP) --recursive $(HELD_OUT_S3)/biological_validation/raw_predictions/ \
	         held_out_projects/biological_validation/raw_predictions/
	$(S3_CP) --recursive $(HELD_OUT_S3)/lcfm/PXD014877_predictions/ \
	         held_out_projects/lcfm/PXD014877_predictions/
	$(S3_CP) --recursive $(HELD_OUT_S3)/acfm/PXD014877_predictions/ \
	         held_out_projects/acfm/PXD014877_predictions/
	$(S3_CP) --recursive $(FASTA_S3)/ fasta/
	@echo "=== Downloading Casanovo data ==="
	mkdir -p analysis_data/casanovo/helaqc_mgf analysis_data/casanovo/celegans_mgf analysis_data/casanovo/predictions
	$(S3_CP) --recursive $(ANALYSIS_S3)/casanovo/helaqc_mgf/    analysis_data/casanovo/helaqc_mgf/
	$(S3_CP) --recursive $(ANALYSIS_S3)/casanovo/celegans_mgf/  analysis_data/casanovo/celegans_mgf/
	$(S3_CP) --recursive $(ANALYSIS_S3)/casanovo/predictions/   analysis_data/casanovo/predictions/
	@echo "=== Downloading PrimeNovo data ==="
	mkdir -p analysis_data/primenovo/helaqc_mgf analysis_data/primenovo/celegans_mgf analysis_data/primenovo/predictions
	$(S3_CP) --recursive $(ANALYSIS_S3)/primenovo/helaqc_mgf/   analysis_data/primenovo/helaqc_mgf/
	$(S3_CP) --recursive $(ANALYSIS_S3)/primenovo/celegans_mgf/ analysis_data/primenovo/celegans_mgf/
	$(S3_CP) --recursive $(ANALYSIS_S3)/primenovo/predictions/  analysis_data/primenovo/predictions/
	@echo "=== download-analysis-data complete ==="

## Upload analysis plots under results/*/plots/ to S3 (run after plot-all).
.PHONY: upload-analysis-plots
upload-analysis-plots:
	@echo "=== Uploading analysis plots ==="
	@for dir in $(ANALYSIS_RESULTS)/*_predictions_*/; do \
		[ -d "$$dir" ] || continue; \
		if [ -d "$$dir/plots" ]; then \
			name=$$(basename "$$dir"); \
			echo "Uploading plots $$name -> $(ANALYSIS_S3)/results/$$name/plots/"; \
			$(S3_CP) --recursive "$$dir/plots/" "$(ANALYSIS_S3)/results/$$name/plots/"; \
		fi; \
	done
	@echo "=== upload-analysis-plots complete ==="

## Upload models to S3 (predictions upload during predict-all; plots via upload-analysis-plots).
upload-analysis-results: upload-analysis-plots
	@echo "=== upload-analysis-results complete (predictions uploaded per predict) ==="

## Download trained models, predictions, and plots from S3 after a cluster run.
download-analysis-results:
	@echo "=== Downloading models ==="
	for m in instanovo_helaqc instanovo_celegans casanovo_helaqc casanovo_celegans primenovo_helaqc primenovo_celegans; do \
		mkdir -p $(ANALYSIS_MODELS)/$$m; \
		$(S3_CP) --recursive $(ANALYSIS_S3)/models/$$m/ $(ANALYSIS_MODELS)/$$m/; \
	done
	@echo "=== Downloading predictions + plots ==="
	mkdir -p $(ANALYSIS_RESULTS)
	$(S3_CP) --recursive $(ANALYSIS_S3)/results/ $(ANALYSIS_RESULTS)/
	@echo "=== download-analysis-results complete ==="

## Full cluster pipeline: download data + pre-trained models, predict (upload
## preds to $(ANALYSIS_S3)/results/<dataset>/ after each), plot, upload plots.
## Intended to be the manifest.yaml entrypoint command.
## Override CASANOVO_ROOT / PRIMENOVO_ROOT to analysis_data/ paths on the
## cluster since the external repos are not present there.
analysis-pipeline: download-analysis-data download-analysis-models predict-all plot-all upload-analysis-plots

# ═══════════════════════════════════════════════════════
# Convenience targets
# ═══════════════════════════════════════════════════════
.PHONY: train-instanovo-split-parquets predict-instanovo-split-parquets plot-instanovo-split-parquets

train-instanovo-split-parquets: train-instanovo-pxd019483 train-instanovo-sbrodae train-instanovo-celegans

predict-instanovo-split-parquets: \
	predict-instanovo-pxd019483-test predict-instanovo-pxd019483-unlabelled \
	predict-instanovo-sbrodae-test predict-instanovo-sbrodae-unlabelled \
	predict-instanovo-celegans-test predict-instanovo-celegans-unlabelled

plot-instanovo-split-parquets: plot-instanovo-pxd019483 plot-instanovo-sbrodae plot-instanovo-celegans

.PHONY: plot-fdr-method-comparison

plot-fdr-method-comparison:
	uv run python scripts/plot_fdr_method_comparison.py \
		--output-dir results/fdr_method_comparison/

.PHONY: train-all predict-all plot-all

train-all: train-instanovo-helaqc train-instanovo-celegans \
           train-casanovo-helaqc train-casanovo-celegans \
           train-primenovo-helaqc train-primenovo-celegans

predict-all: predict-instanovo-helaqc-test predict-instanovo-helaqc-unlabelled predict-instanovo-helaqc-raw_less_train \
             predict-instanovo-celegans-test predict-instanovo-celegans-unlabelled predict-instanovo-celegans-raw_less_train \
             predict-casanovo-helaqc-test predict-casanovo-helaqc-unlabelled predict-casanovo-helaqc-raw_less_train \
             predict-casanovo-celegans-test predict-casanovo-celegans-unlabelled predict-casanovo-celegans-raw_less_train \
             predict-primenovo-helaqc-test predict-primenovo-helaqc-unlabelled predict-primenovo-helaqc-raw_less_train \
             predict-primenovo-celegans-test predict-primenovo-celegans-unlabelled predict-primenovo-celegans-raw_less_train

plot-all: plot-instanovo-helaqc plot-instanovo-celegans \
          plot-casanovo-helaqc plot-casanovo-celegans \
          plot-primenovo-helaqc plot-primenovo-celegans
