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
KOINA_PREDICT_CONSTANTS = +koina.input_constants.collision_energies=27 \
                          +koina.input_constants.fragmentation_types=HCD
# External / Astral datasets carry per-row metadata columns:
KOINA_PREDICT_COLUMNS = +koina.input_columns.collision_energies=collision_energy \
                        +koina.input_columns.fragmentation_types=frag_type

#################################################################################
## Docker build commands																#
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
	$(KOINA_OVERRIDES)
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
		training_matrix_output_path=train_feature_matrices/$$project.parquet \
		$(KOINA_OVERRIDES); \
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
	calibrator.seed=42 \
	$(KOINA_OVERRIDES)


#########################################################
## Hyperparameter optimisation (HPO)
#########################################################

.PHONY: hpo hpo-download-data hpo-upload-model

HPO_TRAIN_FEATURES ?= new_train_shuffled/
HPO_VAL_FEATURES   ?= new_val_feature_matrices/PXD010154/
HPO_N_TRIALS       ?= 100
HPO_TIMEOUT        ?= 43200
HPO_CONFIG         ?= scripts/hpo_config.yaml
HPO_OUTPUT_DIR     ?= hpo_best_model

HPO_MODEL_S3 ?= s3://winnow-g88rh/revisions/new_datasets/models

HPO_TRAIN_S3 ?= s3://winnow-g88rh/revisions/new_datasets/new_train_shuffled
HPO_VAL_S3   ?= s3://winnow-g88rh/revisions/new_datasets/new_val_feature_matrices/PXD010154

## Download feature matrices from S3 for HPO (training + validation)
hpo-download-data:
	mkdir -p $(HPO_TRAIN_FEATURES) $(HPO_VAL_FEATURES)
	aws s3 cp --recursive $(HPO_TRAIN_S3) $(HPO_TRAIN_FEATURES)
	aws s3 cp --recursive $(HPO_VAL_S3) $(HPO_VAL_FEATURES)

## Run Optuna HPO for the calibrator on pre-computed feature matrices
hpo: hpo-download-data
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
	aws s3 cp --recursive $(HPO_OUTPUT_DIR) $(RUN_S3)/$(RUN_TS)/model/
	@echo "Model uploaded to $(RUN_S3)/$(RUN_TS)/model/"

#########################################################
## HPO end-to-end pipeline
#########################################################

.PHONY: hpo-pipeline download-eval-data eval-annotated eval-raw \
        eval-external-labelled eval-external-unlabelled \
        feature-analysis ablation

S3_BASE         ?= s3://winnow-g88rh/revisions/new_datasets
HELD_OUT_S3     ?= $(S3_BASE)/held_out_projects
FASTA_S3        ?= $(S3_BASE)/fasta
RUN_S3          ?= $(S3_BASE)/hpo_runs
RUN_TS          ?= $(shell date -u +%Y%m%dT%H%M%SZ)
PREDS_DIR       ?= predictions/hpo_model
EVAL_PLOTS_DIR  ?= analysis/hpo_eval_plots

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
	aws s3 cp --recursive $(HELD_OUT_S3)/biological_validation/ held_out_projects/biological_validation/
	aws s3 cp --recursive $(FASTA_S3)/ fasta/
	@# Full external projects (lcfm + acfm)
	for project in $(EXTERNAL_FULL_PROJECTS); do \
		aws s3 cp --recursive $(HELD_OUT_S3)/lcfm/$$project/ held_out_projects/lcfm/$$project/; \
		aws s3 cp --recursive $(HELD_OUT_S3)/lcfm/$${project}_predictions/ held_out_projects/lcfm/$${project}_predictions/; \
		aws s3 cp --recursive $(HELD_OUT_S3)/acfm/$$project/ held_out_projects/acfm/$$project/; \
		aws s3 cp --recursive $(HELD_OUT_S3)/acfm/$${project}_predictions/ held_out_projects/acfm/$${project}_predictions/; \
	done
	@# PXD023064 -- only the selected files
	mkdir -p held_out_projects/lcfm/PXD023064 held_out_projects/acfm/PXD023064
	for file in $(PXD023064_FILES); do \
		aws s3 cp $(HELD_OUT_S3)/lcfm/PXD023064/$$file.parquet held_out_projects/lcfm/PXD023064/$$file.parquet; \
		aws s3 cp $(HELD_OUT_S3)/acfm/PXD023064/$$file.parquet held_out_projects/acfm/PXD023064/$$file.parquet; \
	done
	aws s3 cp --recursive $(HELD_OUT_S3)/lcfm/PXD023064_predictions/ held_out_projects/lcfm/PXD023064_predictions/
	aws s3 cp --recursive $(HELD_OUT_S3)/acfm/PXD023064_predictions/ held_out_projects/acfm/PXD023064_predictions/
	@# Astral
	aws s3 cp --recursive --exclude "*.ipc" $(S3_BASE)/astral/labelled/ astral/labelled/
	aws s3 cp --recursive --exclude "*.ipc" $(S3_BASE)/astral/full/ astral/full/
	aws s3 cp --recursive $(S3_BASE)/astral/predictions/ astral/predictions/

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
		$(KOINA_PREDICT_CONSTANTS) \
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
		aws s3 cp --recursive $(PREDS_DIR)/$${project}_annotated/ $(RUN_S3)/$(RUN_TS)/eval_annotated/$${project}_annotated/; \
	done
	aws s3 cp --recursive $(EVAL_PLOTS_DIR)/annotated/ $(RUN_S3)/$(RUN_TS)/eval_annotated/plots/

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
		$(KOINA_PREDICT_CONSTANTS) \
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
		aws s3 cp --recursive $(PREDS_DIR)/$${project}_raw/ $(RUN_S3)/$(RUN_TS)/eval_raw/$${project}_raw/; \
	done
	aws s3 cp --recursive $(EVAL_PLOTS_DIR)/raw/ $(RUN_S3)/$(RUN_TS)/eval_raw/plots/

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
		$(KOINA_PREDICT_COLUMNS) \
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
		$(KOINA_PREDICT_COLUMNS) \
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
		$(KOINA_PREDICT_COLUMNS) \
		fdr_control.fdr_threshold=1.0 \
		fdr_control.confidence_column=calibrated_confidence \
		$(KOINA_OVERRIDES)
	uv run python scripts/plot_eval_results.py \
		--predictions-root $(PREDS_DIR) \
		--projects "$(EXTERNAL_FULL_PROJECTS) PXD023064 Astral" \
		--eval-type labelled \
		--output-dir $(EVAL_PLOTS_DIR)/external_labelled
	for project in $(EXTERNAL_FULL_PROJECTS) PXD023064 Astral; do \
		aws s3 cp --recursive $(PREDS_DIR)/$${project}_labelled/ $(RUN_S3)/$(RUN_TS)/eval_external_labelled/$${project}_labelled/; \
	done
	aws s3 cp --recursive $(EVAL_PLOTS_DIR)/external_labelled/ $(RUN_S3)/$(RUN_TS)/eval_external_labelled/plots/

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
		$(KOINA_PREDICT_COLUMNS) \
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
		$(KOINA_PREDICT_COLUMNS) \
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
		$(KOINA_PREDICT_COLUMNS) \
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
		aws s3 cp --recursive $(PREDS_DIR)/$${project}_unlabelled/ $(RUN_S3)/$(RUN_TS)/eval_external_unlabelled/$${project}_unlabelled/; \
	done
	aws s3 cp --recursive $(EVAL_PLOTS_DIR)/external_unlabelled/ $(RUN_S3)/$(RUN_TS)/eval_external_unlabelled/plots/

## Run feature importance analysis on 3 datasets, upload
feature-analysis:
	mkdir -p analysis/feature_analysis/celegans analysis/feature_analysis/sbrodae analysis/feature_analysis/helaqc
	@# C. elegans (PXD014877) -- labelled lcfm, compute features from raw spectra via Koina
	uv run python scripts/analyze_features.py \
		--model-path $(HPO_OUTPUT_DIR) \
		--data-dir . \
		--output-dir analysis/feature_analysis/celegans \
		--train-spectra held_out_projects/lcfm/PXD014877 \
		--train-preds held_out_projects/lcfm/PXD014877_predictions/PXD014877.csv \
		--test-spectra held_out_projects/lcfm/PXD014877 \
		--test-preds held_out_projects/lcfm/PXD014877_predictions/PXD014877.csv \
		--n-background-samples 200 --n-test-samples 500
	@# S. brodae -- annotated bio-val
	uv run python scripts/analyze_features.py \
		--model-path $(HPO_OUTPUT_DIR) \
		--data-dir . \
		--output-dir analysis/feature_analysis/sbrodae \
		--train-spectra held_out_projects/biological_validation/annotated/dataset-sbrodae-annotated-0000-0001.parquet \
		--train-preds held_out_projects/biological_validation/annotated_predictions/dataset-sbrodae-annotated-0000-0001.csv \
		--test-spectra held_out_projects/biological_validation/annotated/dataset-sbrodae-annotated-0000-0001.parquet \
		--test-preds held_out_projects/biological_validation/annotated_predictions/dataset-sbrodae-annotated-0000-0001.csv \
		--n-background-samples 200 --n-test-samples 500
	@# HeLa QC -- annotated bio-val
	uv run python scripts/analyze_features.py \
		--model-path $(HPO_OUTPUT_DIR) \
		--data-dir . \
		--output-dir analysis/feature_analysis/helaqc \
		--train-spectra held_out_projects/biological_validation/annotated/dataset-helaqc-annotated-0000-0001.parquet \
		--train-preds held_out_projects/biological_validation/annotated_predictions/dataset-helaqc-annotated-0000-0001.csv \
		--test-spectra held_out_projects/biological_validation/annotated/dataset-helaqc-annotated-0000-0001.parquet \
		--test-preds held_out_projects/biological_validation/annotated_predictions/dataset-helaqc-annotated-0000-0001.csv \
		--n-background-samples 200 --n-test-samples 500
	aws s3 cp --recursive analysis/feature_analysis/ $(RUN_S3)/$(RUN_TS)/feature_analysis/

## Run feature ablation study on 4 datasets, upload
ablation:
	uv run python scripts/run_feature_ablations.py \
		--train-features $(HPO_TRAIN_FEATURES) \
		--val-features $(HPO_VAL_FEATURES) \
		--output-dir analysis/hpo_ablation \
		--astral-spectra astral/labelled \
		--astral-predictions astral/predictions/astral_labelled.csv \
		--plot-format both \
		$(if $(KOINA_SERVER_URL),--koina-url $(KOINA_SERVER_URL)) \
		$(if $(filter false,$(KOINA_SSL)),--no-koina-ssl)
	aws s3 cp --recursive analysis/hpo_ablation/ $(RUN_S3)/$(RUN_TS)/ablation/

## Full HPO pipeline: tune, evaluate, analyse, upload
hpo-pipeline: hpo-download-data hpo hpo-upload-model download-eval-data \
              eval-annotated eval-raw eval-external-labelled \
              eval-external-unlabelled feature-analysis ablation

#########################################################
## Re-run eval pipeline with a saved HPO model from S3
#########################################################

.PHONY: hpo-download-model hpo-reeval hpo-reeval-annotated hpo-reeval-raw \
        hpo-reeval-external-labelled hpo-reeval-external-unlabelled \
        hpo-reeval-feature-analysis hpo-reeval-ablation

# S3 path to the HPO run whose model you want to evaluate.
# Override on the command line:
#   make hpo-reeval HPO_RUN_TS=20260514T012035Z
HPO_RUN_TS ?= 20260514T012035Z
HPO_MODEL_S3_PATH ?= $(RUN_S3)/$(HPO_RUN_TS)/model

## Download a saved HPO model from S3
hpo-download-model:
	mkdir -p $(HPO_OUTPUT_DIR)
	aws s3 cp --recursive $(HPO_MODEL_S3_PATH)/ $(HPO_OUTPUT_DIR)/

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
		$(KOINA_PREDICT_CONSTANTS) \
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
		$(KOINA_PREDICT_CONSTANTS) \
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
		$(KOINA_PREDICT_COLUMNS) \
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
		$(KOINA_PREDICT_COLUMNS) \
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
	aws s3 cp $(NEW_TRAIN_S3)/$(1)/ data/train/$(1)/ --recursive
	aws s3 cp $(NEW_TRAIN_S3)/predictions/$(1).csv data/train_predictions/$(1).csv
	@echo "Combining parquet shards in data/train/$(1)/ -> data/train/$(1).parquet"
	uv run python -c "import polars as pl; from pathlib import Path; d = Path('data/train/$(1)'); parts = sorted(d.glob('*.parquet')); print(f'  Found {len(parts)} shard(s), {sum(pl.scan_parquet(p).select(pl.len()).collect().item() for p in parts):,} rows total'); pl.concat([pl.read_parquet(p) for p in parts]).write_parquet(d.parent / '$(1).parquet'); print(f'  Written to data/train/$(1).parquet')"
	uv run winnow compute-features \
		dataset.spectrum_path_or_directory=data/train/$(1).parquet \
		dataset.predictions_path=data/train_predictions/$(1).csv \
		training_matrix_output_path=train_feature_matrices/$(1)/training_matrix.parquet \
		koina.server_url=localhost:8500 koina.ssl=false
	aws s3 cp train_feature_matrices/$(1)/ $(NEW_TRAIN_FEATURES_S3)/$(1)/ --recursive
endef

# Template for normal projects: download directory, compute features per-file.
define compute_features_project_template
compute_features_$(1):
	mkdir -p data/train/$(1)/
	mkdir -p data/train_predictions/
	aws s3 cp $(NEW_TRAIN_S3)/$(1)/ data/train/$(1)/ --recursive
	aws s3 cp $(NEW_TRAIN_S3)/predictions/$(1).csv data/train_predictions/$(1).csv
	uv run winnow compute-features \
		dataset.spectrum_path_or_directory=data/train/$(1)/ \
		dataset.predictions_path=data/train_predictions/$(1).csv \
		training_matrix_output_path=train_feature_matrices/$(1)/training_matrix.parquet \
		koina.server_url=localhost:8500 koina.ssl=false
	aws s3 cp train_feature_matrices/$(1)/ $(NEW_TRAIN_FEATURES_S3)/$(1)/ --recursive
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
	aws s3 cp $(NEW_TRAIN_S3)/$(1)/$(2)/ data/train/$(1)/$(2)/ --recursive
	aws s3 cp $(NEW_TRAIN_S3)/predictions/$(1)_$(2).csv data/train_predictions/$(1)_$(2).csv
	uv run winnow compute-features \
		dataset.spectrum_path_or_directory=data/train/$(1)/$(2)/ \
		dataset.predictions_path=data/train_predictions/$(1)_$(2).csv \
		training_matrix_output_path=train_feature_matrices/$(1)/$(2)/training_matrix.parquet \
		koina.server_url=localhost:8500 koina.ssl=false
	aws s3 cp train_feature_matrices/$(1)/$(2)/ $(NEW_TRAIN_FEATURES_S3)/$(1)/$(2)/ --recursive
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
