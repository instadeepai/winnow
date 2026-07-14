"""Contains classes and functions for PSM rescoring and probability calibration."""

from __future__ import annotations

import importlib
import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from numpy.typing import NDArray
from omegaconf import DictConfig
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader

from winnow.calibration.calibration_features import (
    CalibrationFeatures,
    FeatureDependency,
)
from winnow.calibration.features.utils import validate_model_input_params
from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.datasets.feature_dataset import FeatureDataset
from winnow.utils.koina_intensity_config import (
    resolve_feature_model_inputs,
    strip_runtime_keys_from_feature_config,
)
from winnow.utils.paths import resolve_data_path

logger = logging.getLogger(__name__)


def _deserialize_calibration_feature(
    feature_config: Dict[str, Any],
) -> CalibrationFeatures:
    """Rebuild a feature instance from a saved ``config.json`` entry."""
    feature_config = strip_runtime_keys_from_feature_config(dict(feature_config))
    target = feature_config.pop("_target_")
    module_path, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_path)
    feature_class = getattr(module, class_name)
    return feature_class(**feature_config)


def _apply_koina_constant_overrides(
    model_input_constants: Dict[str, Any],
    model_input_columns: Dict[str, str],
    overrides: Optional[Dict[str, Any]],
) -> None:
    """Merge constant overrides and drop those keys from column mappings."""
    if not overrides:
        return
    for key, value in overrides.items():
        if value is None:
            continue
        model_input_constants[key] = value
        model_input_columns.pop(key, None)


def _apply_koina_column_overrides(
    model_input_constants: Dict[str, Any],
    model_input_columns: Dict[str, str],
    overrides: Optional[Dict[str, str]],
) -> None:
    """Merge column overrides and drop those keys from constants."""
    if not overrides:
        return
    for key, value in overrides.items():
        if value is None:
            continue
        # Constants applied first take precedence; ignore default column entries in
        # the composed config when a constant override is already set for this key.
        if key in model_input_constants:
            continue
        model_input_columns[key] = value
        model_input_constants.pop(key, None)


def _feature_config_for_save(feature: CalibrationFeatures) -> Dict[str, Any]:
    """Serialise a feature for ``config.json``, omitting runtime-only Koina keys."""
    return strip_runtime_keys_from_feature_config(feature.get_config())


def _merge_koina_overrides_into_single_feature(
    feature: Any,
    model_input_constants: Optional[Dict[str, Any]],
    model_input_columns: Optional[Dict[str, str]],
) -> None:
    """Apply Koina constant/column overrides to one feature object."""
    merged_constants = dict(feature.model_input_constants or {})
    merged_columns = dict(feature.model_input_columns or {})
    _apply_koina_constant_overrides(
        merged_constants, merged_columns, model_input_constants
    )
    _apply_koina_column_overrides(merged_constants, merged_columns, model_input_columns)
    resolved_constants, resolved_columns = resolve_feature_model_inputs(
        merged_constants or None, merged_columns or None
    )
    validate_model_input_params(resolved_constants, resolved_columns)
    feature.model_input_constants = resolved_constants
    feature.model_input_columns = resolved_columns


def _load_saved_features_section(
    calibrator: Any, features_section: Dict[str, Any]
) -> None:
    """Populate ``calibrator`` with features from a ``config.json`` ``features`` block."""
    for _name, feature_config in features_section.items():
        calibrator.add_feature(_deserialize_calibration_feature(feature_config))


@dataclass
class TrainingHistory:
    """Epoch-level training metrics from calibrator fitting.

    Attributes:
        train_losses: Training loss at the end of each epoch.
        val_losses: Validation loss per epoch (present only when validation data is used).
        val_accuracies: Validation accuracy per epoch.
        best_epoch: Epoch with the best validation loss (0-indexed), or the last
            epoch if no validation data was provided.
        epochs_trained: Total number of epochs completed.
        final_val_loss: Loss on the full validation set after restoring the best
            weights (only when validation was subsampled for early stopping).
        final_val_accuracy: Accuracy on the full validation set in that case.
    """

    train_losses: List[float] = field(default_factory=list)
    val_losses: Optional[List[float]] = None
    val_accuracies: Optional[List[float]] = None
    best_epoch: int = 0
    epochs_trained: int = 0
    final_val_loss: Optional[float] = None
    final_val_accuracy: Optional[float] = None

    def save(self, path: Union[Path, str]) -> None:
        """Save training history to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Union[Path, str]) -> TrainingHistory:
        """Load training history from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        known = {f.name for f in fields(cls)}
        data = {k: v for k, v in data.items() if k in known}
        return cls(**data)

    def plot(
        self,
        output_path: Optional[Union[Path, str]] = None,
        show: bool = False,
    ) -> None:
        """Plot training and validation curves.

        Args:
            output_path: Path to save the plot image.  Not saved if ``None``.
            show: Whether to call ``plt.show()`` interactively.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        epochs = range(1, len(self.train_losses) + 1)

        ax.plot(epochs, self.train_losses, label="Train Loss", linestyle="-")

        if self.val_losses is not None:
            ax.plot(epochs, self.val_losses, label="Validation Loss", linestyle="--")

        if self.val_accuracies is not None:
            ax2 = ax.twinx()
            ax2.plot(
                epochs, self.val_accuracies, label="Validation Accuracy", linestyle=":"
            )
            ax2.set_ylabel("Validation Accuracy")
            ax2.legend(loc="lower left")

        plt.tight_layout()
        plt.grid(False)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Calibrator Training Progress")
        ax.legend(loc="upper left")

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, bbox_inches="tight")

        if show:
            plt.show()

        plt.close(fig)


class CalibratorNetwork(nn.Module):
    """Feed-forward neural network for probability calibration.

    Args:
        input_dim: Number of input features.
        hidden_dims: Sizes of hidden layers.
        dropout: Dropout rate applied after each hidden layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (scalar per sample)."""
        return self.network(x).squeeze(-1)


class ProbabilityCalibrator:
    """Recalibrates probabilities for de novo peptide sequencing predictions.

    Uses a PyTorch feed-forward network trained on pre-computed calibration
    features.  Models are persisted as ``safetensors`` weights plus a
    ``config.json`` metadata file.
    """

    def __init__(
        self,
        features: Optional[
            Union[
                List[CalibrationFeatures],
                Dict[str, CalibrationFeatures],
                DictConfig,
            ]
        ] = None,
        hidden_dims: Tuple[int, ...] = (128, 64),
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        batch_size: int = 1024,
        n_iter_no_change: int = 10,
        tol: float = 1e-4,
        seed: int = 42,
        val_early_stopping_max_psms: Optional[int] = None,
        val_subsample_seed: Optional[int] = None,
        training_feature_columns: Optional[List[str]] = None,
    ) -> None:
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        if max_epochs < 1:
            raise ValueError(f"max_epochs must be at least 1, got {max_epochs}.")
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.seed = seed
        self.val_early_stopping_max_psms = val_early_stopping_max_psms
        self.val_subsample_seed = val_subsample_seed

        self.feature_dict: Dict[str, CalibrationFeatures] = {}
        self.dependencies: Dict[str, FeatureDependency] = {}
        self.dependency_reference_counter: Dict[str, int] = {}

        self.network: Optional[CalibratorNetwork] = None
        self.feature_mean: Optional[torch.Tensor] = None
        self.feature_std: Optional[torch.Tensor] = None
        # Frozen schema after a successful fit / load. Predict and save use this
        # ordered list so column identity matches the trained network.
        self._fitted_feature_columns: Optional[List[str]] = None
        # Optional pre-fit override: ordered subset of registry columns used for
        # training / extract. Cleared when fitted; restored from checkpoints.
        self._training_feature_columns: Optional[List[str]] = None

        if features is not None:
            if isinstance(features, (dict, DictConfig)):
                self.add_features(list(features.values()))
            else:
                self.add_features(list(features))

        if training_feature_columns is not None:
            self.set_training_feature_columns(training_feature_columns)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def columns(self) -> List[str]:
        """Ordered non-confidence feature columns for the MLP input schema.

        Resolution order:

        1. Frozen fitted schema after ``fit`` / ``fit_from_features`` / ``load``.
        2. Else ``training_feature_columns`` when set (pre-fit subset).
        3. Else the full registry flatten of
           :attr:`~winnow.calibration.features.base.CalibrationFeatures.columns`.

        Confidence is never included; callers prepend it when building matrices.
        """
        if self._fitted_feature_columns is not None:
            return list(self._fitted_feature_columns)
        if self._training_feature_columns is not None:
            return list(self._training_feature_columns)
        return self._registry_feature_columns()

    @property
    def training_feature_columns(self) -> Optional[List[str]]:
        """Pre-fit training column override, or ``None`` for the full registry."""
        if self._training_feature_columns is None:
            return None
        return list(self._training_feature_columns)

    @property
    def feature_names(self) -> List[str]:
        """Names of features registered with the calibrator."""
        return list(self.feature_dict.keys())

    # ------------------------------------------------------------------
    # Static methods
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device() -> torch.device:
        """Auto-detect GPU, warn if falling back to CPU."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
        else:
            logger.warning(
                "No GPU detected, falling back to CPU. "
                "Training may be slow for large datasets."
            )
            device = torch.device("cpu")
        return device

    @staticmethod
    def _reject_legacy_pickle_checkpoint(dir_path: Path) -> None:
        """Raise if ``dir_path`` contains a pre-PyTorch pickle checkpoint."""
        legacy_pickle = dir_path / "calibrator.pkl"
        config_path = dir_path / "config.json"
        if legacy_pickle.exists() and not config_path.exists():
            raise ValueError(
                "Legacy pickle checkpoint format is no longer supported. "
                "The directory contains calibrator.pkl from a pre-PyTorch "
                "version of winnow, but this version expects model.safetensors "
                "and config.json. Retrain the calibrator with the current "
                "version to produce a compatible checkpoint."
            )

    # ------------------------------------------------------------------
    # Class methods (persistence)
    # ------------------------------------------------------------------

    @classmethod
    def save(
        cls,
        calibrator: ProbabilityCalibrator,
        dir_path: Union[Path, str],
    ) -> None:
        """Save calibrator weights and config to a directory.

        Writes ``model.safetensors`` (network weights, feature mean/std) and
        ``config.json`` (architecture, hyperparameters, resolved feature configs).

        Args:
            calibrator: The calibrator to save.
            dir_path: Destination directory.
        """
        if calibrator.network is None:
            raise RuntimeError("Cannot save an unfitted calibrator.")
        assert calibrator.feature_mean is not None
        assert calibrator.feature_std is not None

        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

        tensors: Dict[str, torch.Tensor] = {}
        for name, param in calibrator.network.state_dict().items():
            tensors[f"network.{name}"] = param.cpu()
        tensors["feature_mean"] = calibrator.feature_mean.cpu()
        tensors["feature_std"] = calibrator.feature_std.cpu()

        save_file(tensors, dir_path / "model.safetensors")

        config = {
            "input_dim": int(calibrator.feature_mean.shape[0]),
            "hidden_dims": list(calibrator.hidden_dims),
            "dropout": calibrator.dropout,
            "learning_rate": calibrator.learning_rate,
            "weight_decay": calibrator.weight_decay,
            "max_epochs": calibrator.max_epochs,
            "batch_size": calibrator.batch_size,
            "n_iter_no_change": calibrator.n_iter_no_change,
            "tol": calibrator.tol,
            "seed": calibrator.seed,
            "val_early_stopping_max_psms": calibrator.val_early_stopping_max_psms,
            "val_subsample_seed": calibrator.val_subsample_seed,
            "feature_columns": calibrator.columns,
            "training_feature_columns": calibrator.training_feature_columns,
            "feature_names": calibrator.feature_names,
            "features": {
                name: _feature_config_for_save(feature)
                for name, feature in calibrator.feature_dict.items()
            },
        }
        with open(dir_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info("Saved calibrator to %s", dir_path)

    @classmethod
    def load(
        cls,
        pretrained_model_name_or_path: Union[
            Path, str
        ] = "InstaDeepAI/winnow-general-model",
        cache_dir: Optional[Path] = None,
    ) -> ProbabilityCalibrator:
        """Load a calibrator from a local directory or HuggingFace Hub repo.

        Expects the directory to contain ``model.safetensors`` and
        ``config.json``.

        Args:
            pretrained_model_name_or_path: Local directory path or HuggingFace
                repository identifier.
            cache_dir: Optional cache directory for HuggingFace downloads.

        Returns:
            A fitted ``ProbabilityCalibrator`` ready for prediction.

        Raises:
            FileNotFoundError: If the path cannot be resolved.
            ValueError: If the directory contains a legacy pickle checkpoint.
        """
        dir_path = resolve_data_path(
            str(pretrained_model_name_or_path),
            repo_type="model",
            cache_dir=cache_dir,
        )

        cls._reject_legacy_pickle_checkpoint(dir_path)

        config_path = dir_path / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        calibrator = cls(
            hidden_dims=tuple(config["hidden_dims"]),
            dropout=config["dropout"],
            learning_rate=config.get("learning_rate", 1e-3),
            weight_decay=config.get("weight_decay", 1e-4),
            max_epochs=config.get("max_epochs", 100),
            batch_size=config.get("batch_size", 1024),
            n_iter_no_change=config.get("n_iter_no_change", config.get("patience", 10)),
            tol=config.get("tol", 1e-4),
            seed=config.get("seed", 42),
            val_early_stopping_max_psms=config.get("val_early_stopping_max_psms"),
            val_subsample_seed=config.get("val_subsample_seed"),
        )

        _load_saved_features_section(calibrator, config.get("features", {}))
        saved_columns = config.get("feature_columns")
        if saved_columns:
            calibrator._fitted_feature_columns = list(saved_columns)
            # Mirror fitted schema for round-trip readability of the training override.
            calibrator._training_feature_columns = list(saved_columns)
        elif config.get("training_feature_columns") is not None:
            calibrator.set_training_feature_columns(config["training_feature_columns"])

        tensors = load_file(dir_path / "model.safetensors")

        calibrator.feature_mean = tensors["feature_mean"]
        calibrator.feature_std = tensors["feature_std"]

        input_dim = config["input_dim"]
        calibrator.network = CalibratorNetwork(
            input_dim=input_dim,
            hidden_dims=calibrator.hidden_dims,
            dropout=calibrator.dropout,
        )
        network_state = {
            k.removeprefix("network."): v
            for k, v in tensors.items()
            if k.startswith("network.")
        }
        calibrator.network.load_state_dict(network_state)
        calibrator.network.eval()

        logger.info(
            "Loaded calibrator from %s (input_dim=%d, hidden_dims=%s, features=%s)",
            dir_path,
            input_dim,
            calibrator.hidden_dims,
            list(calibrator.feature_dict.keys()),
        )
        calibrator._validate_loaded_checkpoint(input_dim)
        return calibrator

    def apply_koina_model_input_overrides(
        self,
        model_input_constants: Optional[Dict[str, Any]] = None,
        model_input_columns: Optional[Dict[str, str]] = None,
    ) -> None:
        """Merge inference-time Koina inputs into intensity-based features.

        Used by ``winnow predict`` from the top-level config keys
        ``koina.input_constants`` and ``koina.input_columns`` (same names as
        in ``calibrator.yaml``).

        Updates :attr:`model_input_constants` / :attr:`model_input_columns` on
        features that define them (e.g. fragment match, chimeric). Per-key
        overrides replace any previous constant or column mapping for that key.

        Args:
            model_input_constants: Koina input names mapped to constant values
                tiled across all rows.
            model_input_columns: Koina input names mapped to metadata column
                names for per-row values.
        """
        for feature in self.feature_dict.values():
            if not hasattr(feature, "model_input_constants") or not hasattr(
                feature, "model_input_columns"
            ):
                continue
            _merge_koina_overrides_into_single_feature(
                feature,
                model_input_constants,
                model_input_columns,
            )

    # ------------------------------------------------------------------
    # Public instance methods
    # ------------------------------------------------------------------

    def add_feature(self, feature: CalibrationFeatures) -> None:
        """Register a feature for calibration.

        Args:
            feature: The feature to register.

        Raises:
            KeyError: If a feature with the same name is already present.
        """
        if feature.name not in self.feature_dict:
            self.feature_dict[feature.name] = feature
            for dependency in feature.dependencies:
                if dependency.name in self.dependencies:
                    self.dependency_reference_counter[dependency.name] += 1
                else:
                    self.dependencies[dependency.name] = dependency
                    self.dependency_reference_counter[dependency.name] = 1
        else:
            raise KeyError(f"Feature {feature.name} in feature set.")

    def add_features(self, features: List[CalibrationFeatures]) -> None:
        """Register multiple features."""
        for feature in features:
            self.add_feature(feature)

    def set_training_feature_columns(self, columns: Sequence[str] | None) -> None:
        """Set or clear the pre-fit MLP training column schema.

        When set, :attr:`columns` returns this ordered list (before fit) instead
        of the full registry. Registered :class:`CalibrationFeatures` modules
        still compute every column they declare; only extraction / training
        use the subset.

        Prefer calling this after all features are registered. If the registry
        is still empty, names are stored and checked at fit / extract time.
        If features are already registered, names are validated immediately.

        Args:
            columns: Non-confidence metadata column names in MLP order, or
                ``None`` to use the full registry.

        Raises:
            ValueError: If the calibrator is already fitted, or if ``columns``
                contains names absent from the current registry (when features
                are already registered).
        """
        if self._fitted_feature_columns is not None:
            raise ValueError(
                "Cannot change training_feature_columns after the calibrator "
                "has been fitted or loaded. Use a new ProbabilityCalibrator "
                "instance to train another column subset."
            )
        if columns is None:
            self._training_feature_columns = None
            return

        normalised = [str(name) for name in columns]
        if "confidence" in normalised:
            raise ValueError(
                "training_feature_columns must not include 'confidence'; "
                "confidence is always prepended when building the feature matrix."
            )
        if len(normalised) != len(set(normalised)):
            raise ValueError(
                f"training_feature_columns contains duplicates: {normalised}."
            )
        self._training_feature_columns = normalised
        self._validate_training_feature_columns_against_registry()

    def compute_features(self, dataset: CalibrationDataset) -> None:
        """Run feature dependencies and feature computation on a dataset.

        Mutates ``dataset.metadata`` in place: adds feature columns and may
        drop rows when ``learn_from_missing=False`` on individual features.

        Args:
            dataset: The dataset on which features are computed.
        """
        for dependency in self.dependencies.values():
            dependency.compute(dataset=dataset)

        for feature in self.feature_dict.values():
            feature.prepare(dataset=dataset)
            feature.compute(dataset=dataset)

    def fit(
        self,
        train_dataset: CalibrationDataset,
        val_dataset: Optional[CalibrationDataset] = None,
        progress_bar: bool = True,
    ) -> TrainingHistory:
        """Compute features and train the calibrator.

        This is the primary training entry point.  It runs
        :meth:`compute_features` on the supplied datasets, extracts the
        registered feature columns and labels, and trains the calibrator network.

        Both ``train_dataset`` and ``val_dataset`` are mutated in place
        (feature columns are added, and rows may be dropped when individual
        features have ``learn_from_missing=False``).

        Args:
            train_dataset: Labelled training dataset.
            val_dataset: Optional labelled validation dataset for early
                stopping. Subsampling for large sets matches
                :meth:`fit_from_features` (see ``val_early_stopping_max_psms``).
            progress_bar: Whether to display a progress bar during training.

        Returns:
            Epoch-level training metrics.
        """
        self._validate_training_feature_columns_against_registry()
        self.compute_features(train_dataset)
        train_fd = self.to_feature_dataset(train_dataset)

        val_fd = None
        if val_dataset is not None:
            self.compute_features(val_dataset)
            val_fd = self.to_feature_dataset(val_dataset)

        return self._fit_from_features(train_fd, val_fd, progress_bar=progress_bar)

    def to_feature_dataset(self, dataset: CalibrationDataset) -> FeatureDataset:
        """Build a supervised FeatureDataset from labelled featurised metadata.

        Must be called *after* :meth:`compute_features` has populated the
        feature columns on ``dataset.metadata`` (or when those columns are
        already present). Requires a ``correct`` label column.

        Args:
            dataset: Labelled dataset whose metadata already contains the
                feature columns and ``correct``.

        Returns:
            FeatureDataset with float32 feature and label tensors and
            ``columns=list(self.columns)``. Layout is
            ``[confidence, *self.columns]``.
        """
        self._validate_training_feature_columns_against_registry()
        features, labels = self._extract_feature_matrix(dataset, labelled=True)
        return FeatureDataset(
            features=np.asarray(features),
            labels=np.asarray(labels),
            columns=list(self.columns),
        )

    def fit_from_features(
        self,
        train_dataset: FeatureDataset,
        val_dataset: Optional[FeatureDataset] = None,
        progress_bar: bool = True,
        epoch_callback: Optional[Callable[[int, float], None]] = None,
    ) -> TrainingHistory:
        """Train the calibrator from pre-computed feature arrays.

        Use this for the two-phase workflow where features have already been
        computed and saved as Parquet, reloaded with
        :meth:`FeatureDataset.from_parquet`, then aligned via
        :meth:`FeatureDataset.select_for` if necessary so
        ``train_dataset.columns == list(calibrator.columns)``.

        Args:
            train_dataset: Training features and labels whose ``columns`` must
                exactly match :attr:`columns` (names and order).
            val_dataset: Optional held-out validation set for early stopping.
                When it has more than ``val_early_stopping_max_psms`` rows, a
                random subset (``val_subsample_seed``, defaulting to ``seed``)
                is evaluated each epoch; after training, metrics on the full
                validation set are stored in ``TrainingHistory.final_val_*``.
                Must have the same ``columns`` as ``train_dataset``.
            progress_bar: Whether to display a progress bar during training.
            epoch_callback: Optional callback invoked after each validation
                step with ``(epoch, val_loss)``.  Useful for external
                hyperparameter optimisation frameworks (e.g. Optuna pruning).
                Only called when *val_dataset* is provided.

        Returns:
            Epoch-level training metrics.

        Raises:
            ValueError: If train (or validation) feature column names/order
                do not match the calibrator schema.
        """
        return self._fit_from_features(
            train_dataset, val_dataset, progress_bar, epoch_callback
        )

    @torch.no_grad()
    def predict(self, dataset: CalibrationDataset) -> None:
        """Predict calibrated probabilities and store them on the dataset.

        The ``calibrated_confidence`` column is added to ``dataset.metadata``.

        Args:
            dataset: The CalibrationDataset containing the features for which predictions are made.
        """
        if self.network is None:
            raise RuntimeError(
                "Calibrator has not been fitted or loaded. Call fit() or load() first."
            )
        assert self.feature_mean is not None
        assert self.feature_std is not None

        if len(dataset) == 0:
            raise ValueError(
                "Cannot predict on an empty dataset: no spectra remain after "
                "feature computation. Check feature validity filters and "
                "learn_from_missing settings."
            )

        self._validate_fitted_feature_dimensions()
        features = self._extract_feature_matrix(dataset, labelled=False)
        probs = self._predict_feature_matrix(features)
        dataset.metadata["calibrated_confidence"] = probs.tolist()

    def remove_feature(self, name: str) -> None:
        """Remove a feature and its orphaned dependencies."""
        feature = self.feature_dict.pop(name)
        for dependency in feature.dependencies:
            self.dependency_reference_counter[dependency.name] -= 1
            if self.dependency_reference_counter[dependency.name] == 0:
                self.dependency_reference_counter.pop(dependency.name)
                self.dependencies.pop(dependency.name)

    # ------------------------------------------------------------------
    # Private instance methods
    # ------------------------------------------------------------------

    def _registry_feature_columns(self) -> List[str]:
        """Column names from currently registered features, in registry order."""
        return [
            column
            for feature in self.feature_dict.values()
            for column in feature.columns
        ]

    def _validate_training_feature_columns_against_registry(self) -> None:
        """Raise if the training override lists names absent from the registry."""
        if self._training_feature_columns is None or not self.feature_dict:
            return
        registry = self._registry_feature_columns()
        registry_set = set(registry)
        unknown = [
            name for name in self._training_feature_columns if name not in registry_set
        ]
        if unknown:
            raise ValueError(
                f"training_feature_columns contains unknown column(s) {unknown}. "
                f"Available registry columns: {registry}."
            )

    def _validate_fitted_subset_of_registry(self) -> None:
        """Raise if fitted columns are not a subset of the live registry."""
        if self._fitted_feature_columns is None or not self.feature_dict:
            return
        registry_set = set(self._registry_feature_columns())
        missing = [
            name for name in self._fitted_feature_columns if name not in registry_set
        ]
        if missing:
            raise ValueError(
                "Calibrator feature schema mismatch: fitted feature column(s) "
                f"{missing} are not produced by the registered features "
                f"(registry columns: {self._registry_feature_columns()}). "
                "Retrain the calibrator or restore the feature modules that "
                "produced these columns."
            )

    def _extract_feature_matrix(
        self, dataset: CalibrationDataset, labelled: bool
    ) -> Union[
        NDArray[np.float64],
        Tuple[NDArray[np.float64], NDArray[np.float64]],
    ]:
        """Pull registered feature columns (and optionally labels) from metadata.

        Selects ``confidence`` followed by :attr:`columns` by name.
        Must be called *after* :meth:`compute_features` has populated the
        feature columns on ``dataset.metadata``.

        Args:
            dataset: Dataset whose metadata already contains the feature
                columns.
            labelled: If ``True``, also return the ``correct`` column as
                labels.

        Returns:
            Feature matrix, or ``(features, labels)`` when
            ``labelled=True``.
        """
        feature_columns = [dataset.confidence_column]
        feature_columns.extend(self.columns)
        features = dataset.metadata[feature_columns]

        if labelled:
            labels = dataset.metadata["correct"]
            return features.values, labels.values
        return features.values

    def _predict_feature_matrix(
        self, features: NDArray[np.float64]
    ) -> NDArray[np.float32]:
        """Run batched network inference on a host feature matrix.

        Args:
            features: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Calibrated probabilities as a float32 array of shape ``(n_samples,)``.
        """
        assert self.network is not None
        assert self.feature_mean is not None
        assert self.feature_std is not None

        device = next(self.network.parameters()).device
        n = len(features)
        probs = np.empty(n, dtype=np.float32)
        self.network.eval()
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            x = torch.as_tensor(features[start:end], dtype=torch.float32, device=device)
            x = (x - self.feature_mean) / self.feature_std
            probs[start:end] = torch.sigmoid(self.network(x)).cpu().numpy()
        return probs

    def _early_stopping_validation_split(
        self,
        val_dataset: FeatureDataset,
    ) -> Tuple[FeatureDataset, bool]:
        """Return validation data for early stopping, optionally subsampled."""
        max_psms = self.val_early_stopping_max_psms
        n_val = len(val_dataset)
        if max_psms is None or max_psms <= 0 or n_val <= max_psms:
            return val_dataset, False

        subsample_seed = (
            self.val_subsample_seed
            if self.val_subsample_seed is not None
            else self.seed
        )
        rng = np.random.default_rng(subsample_seed)
        idx = rng.permutation(n_val)[:max_psms]
        val_early = FeatureDataset(
            features=val_dataset.features[idx].cpu().numpy(),
            labels=val_dataset.labels[idx].cpu().numpy(),
            columns=val_dataset.columns,
        )
        logger.info(
            "Using %d of %d validation PSMs for early stopping "
            "(val_early_stopping_max_psms=%s, val_subsample_seed=%s).",
            len(val_early),
            n_val,
            max_psms,
            subsample_seed,
        )
        return val_early, True

    def _record_full_validation_if_subsampled(
        self,
        val_was_subsampled: bool,
        val_dataset: Optional[FeatureDataset],
        history: TrainingHistory,
        criterion: nn.Module,
        device: torch.device,
    ) -> None:
        """After training, evaluate on full validation when early stopping used a subset."""
        if not val_was_subsampled or val_dataset is None:
            return
        assert self.network is not None
        final_loss, final_acc = self._evaluate(val_dataset, criterion, device)
        history.final_val_loss = final_loss
        history.final_val_accuracy = final_acc
        logger.info(
            "Full validation set metrics after training (n=%d): "
            "loss=%.6f, accuracy=%.4f",
            len(val_dataset),
            final_loss,
            final_acc,
        )

    def _fit_from_features(
        self,
        train_dataset: FeatureDataset,
        val_dataset: Optional[FeatureDataset] = None,
        progress_bar: bool = True,
        epoch_callback: Optional[Callable[[int, float], None]] = None,
    ) -> TrainingHistory:
        """Train the calibrator network on pre-computed features.

        Args:
            train_dataset: Training features and labels.
            val_dataset: Optional held-out validation set for early stopping.
            progress_bar: Whether to display a progress bar during training.
            epoch_callback: Optional callback invoked after each validation
                step with ``(epoch, val_loss)``.  Only called when
                *val_dataset* is provided.

        Returns:
            Epoch-level training metrics.

        Raises:
            ValueError: If feature column names/order do not match the schema.
        """
        torch.manual_seed(self.seed)
        device = self._resolve_device()

        self._validate_training_feature_columns_against_registry()
        self._validate_feature_dataset_schema(train_dataset, context="training")
        if val_dataset is not None:
            self._validate_train_val_feature_columns(train_dataset, val_dataset)
            self._validate_feature_dataset_schema(val_dataset, context="validation")

        input_dim = train_dataset.features.shape[1]

        self.network = CalibratorNetwork(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(device)

        self.feature_mean = train_dataset.features.mean(dim=0).to(device)
        self.feature_std = train_dataset.features.std(dim=0).clamp(min=1e-8).to(device)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss()

        val_early: Optional[FeatureDataset] = None
        val_was_subsampled = False
        if val_dataset is not None:
            val_early, val_was_subsampled = self._early_stopping_validation_split(
                val_dataset
            )

        has_val = val_early is not None
        history = TrainingHistory(
            val_losses=[] if has_val else None,
            val_accuracies=[] if has_val else None,
        )
        best_val_loss = float("inf")
        best_state: Optional[Dict[str, torch.Tensor]] = None
        epochs_without_improvement = 0

        for epoch in tqdm(
            range(self.max_epochs),
            disable=not progress_bar,
            desc="Training calibrator",
            unit="epoch",
        ):
            avg_train_loss = self._train_one_epoch(
                train_loader,
                optimizer,
                criterion,
                device,
            )
            history.train_losses.append(avg_train_loss)

            if val_early is None:
                history.best_epoch = epoch
                continue

            should_stop, best_val_loss, best_state, epochs_without_improvement = (
                self._validation_step(
                    val_early,
                    criterion,
                    device,
                    history,
                    epoch,
                    best_val_loss,
                    best_state,
                    epochs_without_improvement,
                )
            )

            if epoch_callback is not None:
                assert history.val_losses is not None
                epoch_callback(epoch, history.val_losses[-1])

            if should_stop:
                break

        history.epochs_trained = len(history.train_losses)

        if best_state is not None:
            self.network.load_state_dict(best_state)
            self.network.to(device)

        self._record_full_validation_if_subsampled(
            val_was_subsampled,
            val_dataset,
            history,
            criterion,
            device,
        )

        # Freeze the schema that matched this successful fit (registry or a
        # pre-set fitted override).
        self._fitted_feature_columns = list(self.columns)
        return history

    @torch.no_grad()
    def _evaluate(
        self,
        dataset: FeatureDataset,
        criterion: nn.Module,
        device: torch.device,
    ) -> Tuple[float, float]:
        """Compute loss and accuracy on a dataset."""
        assert self.network is not None
        assert self.feature_mean is not None
        assert self.feature_std is not None

        self.network.eval()
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        total_loss = 0.0
        correct = 0
        total = 0
        for features, labels in loader:
            features = (features.to(device) - self.feature_mean) / self.feature_std
            labels = labels.to(device)
            logits = self.network(features)
            total_loss += criterion(logits, labels).item() * len(labels)
            preds = (logits > 0.0).float()
            correct += (preds == labels).sum().item()
            total += len(labels)

        return total_loss / max(total, 1), correct / max(total, 1)

    def _snapshot_state(self) -> Dict[str, torch.Tensor]:
        """Return a CPU copy of the current network state dict."""
        assert self.network is not None
        return {k: v.cpu().clone() for k, v in self.network.state_dict().items()}

    def _train_one_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
    ) -> float:
        """Run one training epoch, return the average loss."""
        assert self.network is not None
        assert self.feature_mean is not None
        assert self.feature_std is not None

        self.network.train()
        epoch_loss = 0.0
        n_batches = 0
        for features, labels in train_loader:
            features = (features.to(device) - self.feature_mean) / self.feature_std
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = self.network(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        return epoch_loss / max(n_batches, 1)

    def _validation_step(
        self,
        val_dataset: FeatureDataset,
        criterion: nn.Module,
        device: torch.device,
        history: TrainingHistory,
        epoch: int,
        best_val_loss: float,
        best_state: Optional[Dict[str, torch.Tensor]],
        epochs_without_improvement: int,
    ) -> Tuple[bool, float, Optional[Dict[str, torch.Tensor]], int]:
        """Evaluate on validation set and check early stopping.

        Stops after ``n_iter_no_change`` consecutive epochs where validation loss
        did not improve by at least ``tol`` over the running best (sklearn-style).

        Returns:
            Tuple of (should_stop, best_val_loss, best_state,
            epochs_without_improvement).
        """
        val_loss, val_acc = self._evaluate(val_dataset, criterion, device)

        assert history.val_losses is not None
        assert history.val_accuracies is not None
        history.val_losses.append(val_loss)
        history.val_accuracies.append(val_acc)

        # Match sklearn SGD early stopping: count epochs where loss did not beat
        # ``best_val_loss - tol`` (see ``loss > best_loss - tol`` in sklearn).
        if val_loss <= best_val_loss - self.tol:
            best_val_loss = val_loss
            best_state = self._snapshot_state()
            history.best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= self.n_iter_no_change:
            return True, best_val_loss, best_state, epochs_without_improvement

        return False, best_val_loss, best_state, epochs_without_improvement

    def _feature_input_dim(self) -> int:
        """Return the number of model inputs (confidence plus feature columns)."""
        return 1 + len(self.columns)

    def _validate_feature_dataset_schema(
        self,
        dataset: FeatureDataset,
        *,
        context: str,
    ) -> None:
        """Raise if a FeatureDataset schema does not match calibrator.columns."""
        expected = list(self.columns)
        actual = list(dataset.columns)
        if actual == expected:
            return

        missing = [name for name in expected if name not in actual]
        extras = [name for name in actual if name not in expected]
        parts = [
            f"{context.capitalize()} FeatureDataset.columns {actual} do not "
            f"match calibrator.columns {expected} "
            f"(lengths {len(actual)} vs {len(expected)})."
        ]
        if missing:
            parts.append(f"Missing: {missing}.")
        if extras:
            parts.append(f"Extra: {extras}.")
        if not missing and not extras:
            parts.append("Column names match but order differs.")
        parts.append(
            "Align with FeatureDataset.select_for(calibrator) before fit_from_features."
        )
        raise ValueError(" ".join(parts))

    def _validate_train_val_feature_columns(
        self,
        train_dataset: FeatureDataset,
        val_dataset: FeatureDataset,
    ) -> None:
        """Raise if train and validation FeatureDatasets disagree on columns."""
        train_cols = list(train_dataset.columns)
        val_cols = list(val_dataset.columns)
        if train_cols == val_cols:
            return
        raise ValueError(
            "Training and validation FeatureDataset.columns must be identical "
            f"(names and order). Train: {train_cols}. Validation: {val_cols}. "
            "Load both from compatible Parquets and call "
            "select_for(calibrator) on each before fit_from_features."
        )

    def _validate_loaded_checkpoint(self, input_dim: int) -> None:
        """Raise if saved weights and config feature metadata are inconsistent."""
        if self.feature_mean is None:
            return

        trained_dim = int(self.feature_mean.shape[0])
        if trained_dim != input_dim:
            raise ValueError(
                "Calibrator checkpoint is corrupt: config input_dim "
                f"({input_dim}) does not match saved normalisation stats "
                f"({trained_dim} feature(s))."
            )

        if self._fitted_feature_columns is not None:
            expected_dim = 1 + len(self._fitted_feature_columns)
            if trained_dim != expected_dim:
                raise ValueError(
                    "Calibrator feature dimension mismatch: the saved model "
                    f"was trained with {trained_dim} input feature(s) but the "
                    "checkpoint lists "
                    f"{len(self._fitted_feature_columns)} feature column(s) "
                    f"({self._fitted_feature_columns}). "
                    "This usually means the calibrator checkpoint is "
                    "incompatible with the installed version of winnow. "
                    "Retrain the calibrator or use a checkpoint published "
                    "for this version."
                )

            self._validate_fitted_subset_of_registry()

    def _validate_fitted_feature_dimensions(self) -> None:
        """Raise if the current feature set does not match the fitted network."""
        if self.feature_mean is None:
            return

        trained_dim = int(self.feature_mean.shape[0])
        expected_dim = self._feature_input_dim()
        if trained_dim != expected_dim:
            raise ValueError(
                "Calibrator feature dimension mismatch: the saved model was trained "
                f"with {trained_dim} input feature(s) but the current feature set "
                f"produces {expected_dim} (confidence plus {len(self.columns)} "
                f"feature column(s): {self.columns}). "
                "This usually means the calibrator checkpoint is incompatible with "
                "the installed version of winnow. Retrain the calibrator or use a "
                "checkpoint published for this version."
            )

        self._validate_fitted_subset_of_registry()
