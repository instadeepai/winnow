"""Contains classes and functions for PSM rescoring and probability calibration."""

from __future__ import annotations

import importlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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
from winnow.datasets.calibration_dataset import CalibrationDataset
from winnow.datasets.feature_dataset import FeatureDataset
from winnow.utils.paths import resolve_data_path

logger = logging.getLogger(__name__)


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
    """

    train_losses: List[float] = field(default_factory=list)
    val_losses: Optional[List[float]] = None
    val_accuracies: Optional[List[float]] = None
    best_epoch: int = 0
    epochs_trained: int = 0

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

        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(self.train_losses) + 1)

        ax.plot(
            epochs,
            self.train_losses,
            label="Train Loss",
            color="#2563eb",
            linewidth=2,
        )

        if self.val_losses is not None:
            ax.plot(
                epochs,
                self.val_losses,
                label="Val Loss",
                color="#dc2626",
                linewidth=2,
                linestyle="--",
            )
            ax.axvline(
                x=self.best_epoch + 1,
                color="#059669",
                linestyle=":",
                label=f"Best epoch ({self.best_epoch + 1})",
            )

        if self.val_accuracies is not None:
            ax2 = ax.twinx()
            ax2.plot(
                epochs,
                self.val_accuracies,
                label="Val Accuracy",
                color="#7c3aed",
                linewidth=2,
                linestyle=":",
            )
            ax2.set_ylabel("Validation Accuracy", color="#7c3aed", fontsize=12)
            ax2.tick_params(axis="y", labelcolor="#7c3aed")
            ax2.legend(loc="center right")

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Calibrator Training Progress", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")

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
        patience: int = 10,
        seed: int = 42,
    ) -> None:
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.seed = seed

        self.feature_dict: Dict[str, CalibrationFeatures] = {}
        self.dependencies: Dict[str, FeatureDependency] = {}
        self.dependency_reference_counter: Dict[str, int] = {}

        self.network: Optional[CalibratorNetwork] = None
        self.feature_mean: Optional[torch.Tensor] = None
        self.feature_std: Optional[torch.Tensor] = None

        if features is not None:
            if isinstance(features, (dict, DictConfig)):
                self.add_features(list(features.values()))
            else:
                self.add_features(list(features))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def columns(self) -> List[str]:
        """Column names corresponding to the features added to the calibrator."""
        return [
            column
            for feature in self.feature_dict.values()
            for column in feature.columns
        ]

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
            "patience": calibrator.patience,
            "seed": calibrator.seed,
            "feature_columns": calibrator.columns,
            "feature_names": calibrator.feature_names,
            "features": {
                name: feature.get_config()
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
        """
        dir_path = resolve_data_path(
            str(pretrained_model_name_or_path),
            repo_type="model",
            cache_dir=cache_dir,
        )

        with open(dir_path / "config.json") as f:
            config = json.load(f)

        calibrator = cls(
            hidden_dims=tuple(config["hidden_dims"]),
            dropout=config["dropout"],
            learning_rate=config.get("learning_rate", 1e-3),
            weight_decay=config.get("weight_decay", 1e-4),
            max_epochs=config.get("max_epochs", 100),
            batch_size=config.get("batch_size", 1024),
            patience=config.get("patience", 10),
            seed=config.get("seed", 42),
        )

        for _feat_name, feat_cfg in config.get("features", {}).items():
            feat_cfg = dict(feat_cfg)
            target = feat_cfg.pop("_target_")
            module_path, class_name = target.rsplit(".", 1)
            module = importlib.import_module(module_path)
            feat_cls = getattr(module, class_name)
            calibrator.add_feature(feat_cls(**feat_cfg))

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
        return calibrator

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

    def compute_features(
        self, dataset: CalibrationDataset, labelled: bool
    ) -> Union[
        NDArray[np.float64],
        Tuple[NDArray[np.float64], NDArray[np.float64]],
    ]:
        """Compute calibration features from a CalibrationDataset.

        Args:
            dataset: The dataset from which features are computed.
            labelled: Whether to also return labels.

        Returns:
            Feature matrix, or ``(features, labels)`` when ``labelled=True``.
        """
        for dependency in self.dependencies.values():
            dependency.compute(dataset=dataset)

        for feature in self.feature_dict.values():
            feature.prepare(dataset=dataset)
            feature.compute(dataset=dataset)

        feature_columns = [dataset.confidence_column]
        feature_columns.extend(self.columns)
        features = dataset.metadata[feature_columns]

        if labelled:
            labels = dataset.metadata["correct"]
            return features.values, labels.values
        else:
            return features.values

    def fit(
        self,
        train_dataset: FeatureDataset,
        val_dataset: Optional[FeatureDataset] = None,
        progress_bar: bool = True,
    ) -> TrainingHistory:
        """Train the calibrator network on pre-computed features.

        Args:
            train_dataset: Training features and labels.
            val_dataset: Optional held-out validation set for early stopping.

        Returns:
            Epoch-level training metrics.
        """
        torch.manual_seed(self.seed)
        device = self._resolve_device()

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

        has_val = val_dataset is not None
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

            if val_dataset is None:
                history.best_epoch = epoch
                continue

            should_stop, best_val_loss, best_state, epochs_without_improvement = (
                self._validation_step(
                    val_dataset,
                    criterion,
                    device,
                    history,
                    epoch,
                    avg_train_loss,
                    best_val_loss,
                    best_state,
                    epochs_without_improvement,
                )
            )
            if should_stop:
                break

        history.epochs_trained = epoch + 1  # noqa: F821 — loop always runs

        if best_state is not None:
            self.network.load_state_dict(best_state)
            self.network.to(device)

        return history

    @torch.no_grad()
    def predict(self, dataset: CalibrationDataset) -> None:
        """Predict calibrated probabilities and store them on the dataset.

        The ``calibrated_confidence`` column is added to ``dataset.metadata``.

        Args:
            dataset: The dataset for which predictions are made.
        """
        if self.network is None:
            raise RuntimeError(
                "Calibrator has not been fitted or loaded. Call fit() or load() first."
            )
        assert self.feature_mean is not None
        assert self.feature_std is not None

        features = self.compute_features(dataset=dataset, labelled=False)

        device = next(self.network.parameters()).device
        x = torch.as_tensor(features, dtype=torch.float32, device=device)
        x = (x - self.feature_mean) / self.feature_std

        self.network.eval()
        logits = self.network(x)
        probs = torch.sigmoid(logits).cpu().numpy()

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
        avg_train_loss: float,
        best_val_loss: float,
        best_state: Optional[Dict[str, torch.Tensor]],
        epochs_without_improvement: int,
    ) -> Tuple[bool, float, Optional[Dict[str, torch.Tensor]], int]:
        """Evaluate on validation set and check early stopping.

        Returns:
            Tuple of (should_stop, best_val_loss, best_state,
            epochs_without_improvement).
        """
        val_loss, val_acc = self._evaluate(val_dataset, criterion, device)

        assert history.val_losses is not None
        assert history.val_accuracies is not None
        history.val_losses.append(val_loss)
        history.val_accuracies.append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = self._snapshot_state()
            history.best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        logger.info(
            "Epoch %d/%d  train_loss=%.5f  val_loss=%.5f  val_acc=%.4f",
            epoch + 1,
            self.max_epochs,
            avg_train_loss,
            val_loss,
            val_acc,
        )

        if epochs_without_improvement >= self.patience:
            logger.info(
                "Early stopping at epoch %d (patience=%d).",
                epoch + 1,
                self.patience,
            )
            return True, best_val_loss, best_state, epochs_without_improvement

        return False, best_val_loss, best_state, epochs_without_improvement
