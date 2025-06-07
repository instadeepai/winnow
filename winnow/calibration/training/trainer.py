"""Interfaces for the Trainer."""

import dataclasses
import functools
from typing import Callable, Dict

import jax
import jax.numpy as jnp

import orbax.checkpoint as ocp

from flax import nnx

import optax

from jaxtyping import Float
from ray.data import Dataset
from tqdm import tqdm

from winnow.calibration.training.loggers import LoggingManager
from winnow.calibration.training.data import pad_batch_spectra


@dataclasses.dataclass
class Metrics:
    """Metrics for the model."""

    metrics: Dict[str, float]
    checkpoint_metric: str
    checkpoint_metric_value: float


@dataclasses.dataclass
class MetricsManager:
    """Manager that logs metrics to the console and Neptune."""

    metrics: Dict[
        str,
        Callable[
            [
                Float[jnp.ndarray, " batch_size ..."],
                Float[jnp.ndarray, " batch_size ..."],
            ],
            float,
        ],
    ]
    checkpoint_metric: str

    def evaluate(
        self,
        y_pred: Float[jnp.ndarray, " batch_size ..."],
        y_true: Float[jnp.ndarray, " batch_size ..."],
    ) -> Metrics:
        """Evaluate the model.

        Args:
            y_true: The true labels.
            y_pred: The predicted labels.

        Returns:
            metrics: The metrics for the model.
            checkpoint_metric: The checkpoint metric for the model.
        """
        metrics = {}
        for metric_name, metric_fn in self.metrics.items():
            metrics[metric_name] = metric_fn(y_true, y_pred)
        checkpoint_metric = metrics[self.checkpoint_metric]
        return Metrics(
            metrics=metrics,
            checkpoint_metric=self.checkpoint_metric,
            checkpoint_metric_value=checkpoint_metric,
        )


@dataclasses.dataclass
class TrainerConfig:
    """Configuration for the trainer."""

    learning_rate: float
    num_epochs: int
    batch_size: int
    eval_frequency: int
    eval_batch_size: int
    patience: int
    num_peaks: int


class Trainer:
    """Trainer that trains a model."""

    def __init__(
        self,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        logging_manager: LoggingManager,
        checkpoint_manager: ocp.CheckpointManager,
        metrics_manager: MetricsManager,
    ):
        self.model = model
        self.optimizer = optimizer
        self.logging_manager = logging_manager
        self.checkpoint_manager = checkpoint_manager
        self.metrics_manager = metrics_manager
        self.graphdefs, self.state = nnx.split((self.model, self.optimizer))

        def loss_fn(
            model: nnx.Module,
            mz_array: jnp.ndarray,
            intensity_array: jnp.ndarray,
            spectrum_mask: jnp.ndarray,
            residue_indices: jnp.ndarray,
            modification_indices: jnp.ndarray,
            peptide_mask: jnp.ndarray,
            labels: jnp.ndarray,
        ) -> jnp.ndarray:
            """Loss function."""
            logits = model(
                mz_array=mz_array,
                intensity_array=intensity_array,
                spectrum_mask=spectrum_mask,
                residue_indices=residue_indices,
                modification_indices=modification_indices,
                peptide_mask=peptide_mask,
            )  # type: ignore[operator]
            return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))

        def update(
            graphdefs: nnx.GraphDef,
            state: nnx.State,
            mz_array: jnp.ndarray,
            intensity_array: jnp.ndarray,
            spectrum_mask: jnp.ndarray,
            residue_indices: jnp.ndarray,
            modification_indices: jnp.ndarray,
            peptide_mask: jnp.ndarray,
            labels: jnp.ndarray,
        ) -> tuple[float, nnx.State]:
            """Update the model."""
            model, optimizer = nnx.merge(graphdefs, state)
            loss, grads = nnx.value_and_grad(loss_fn)(
                model=model,
                mz_array=mz_array,
                intensity_array=intensity_array,
                spectrum_mask=spectrum_mask,
                residue_indices=residue_indices,
                modification_indices=modification_indices,
                peptide_mask=peptide_mask,
                labels=labels,
            )
            optimizer.update(grads)
            state = nnx.state((model, optimizer))
            return loss, state

        self.update = update

    def __str__(self) -> str:
        return f"""
Trainer(
    model={self.model},
    optimizer={self.optimizer},
    logging_manager={self.logging_manager},
    checkpoint_manager={self.checkpoint_manager},
    metrics_manager={self.metrics_manager}
)
        """

    def train(
        self, train_data: Dataset, val_data: Dataset, config: TrainerConfig
    ) -> None:
        """Train the model.

        Args:
            train_data: The dataset to train on.
            val_data: The dataset to validate on.
            config: The configuration for the trainer.
        """
        step, evals_without_improvement, best_checkpoint_metric = 0, 0, float("-inf")
        for _ in tqdm(range(config.num_epochs), desc="Epochs"):
            for batch in tqdm(
                train_data.iter_batches(
                    prefetch_batches=20,
                    batch_size=config.batch_size,
                    _collate_fn=functools.partial(
                        pad_batch_spectra, num_peaks=config.num_peaks
                    ),  # type: ignore[arg-type]
                ),
                desc="Batches",
            ):
                mz_array = jnp.array(batch["mz_array"])
                intensity_array = jnp.array(batch["intensity_array"])
                spectrum_mask = jnp.array(batch["spectrum_mask"])
                residue_indices = jnp.array(batch["residues_ids"])
                modification_indices = jnp.array(batch["modifications_ids"])
                peptide_mask = jnp.array(batch["peptide_mask"])
                labels = jnp.array(batch["correct"])
                loss, self.state = self.update(
                    graphdefs=self.graphdefs,
                    state=self.state,
                    mz_array=mz_array,
                    intensity_array=intensity_array,
                    spectrum_mask=spectrum_mask,
                    residue_indices=residue_indices,
                    modification_indices=modification_indices,
                    peptide_mask=peptide_mask,
                    labels=labels,
                )
                self.logging_manager.log_metric("train/loss", loss, step)
                step += 1
                if step % config.eval_frequency == 0:
                    checkpoint_metric = self.evaluate(
                        data=val_data,
                        num_peaks=config.num_peaks,
                        batch_size=config.eval_batch_size,
                        step=step,
                    )
                    if checkpoint_metric > best_checkpoint_metric:
                        evals_without_improvement = 0
                        best_checkpoint_metric = checkpoint_metric
                        self.checkpoint_manager.save(
                            step, args=ocp.args.StandardSave(self.state)
                        )
                    else:
                        evals_without_improvement += 1
                    if evals_without_improvement >= config.patience:
                        break
        nnx.update(
            (self.model, self.optimizer), self.state
        )  # Ensure the final state is updated in the graphdefs
        self.logging_manager.log_metric(
            "train/best_checkpoint_metric", best_checkpoint_metric, step
        )

    def evaluate(
        self, data: Dataset, num_peaks: int, batch_size: int, step: int
    ) -> float:
        """Evaluate the model.

        Args:
            data: The dataset to predict on.
            num_peaks: The number of peaks to consider.
            batch_size: The batch size.
            step: The current step.
        """
        model, _ = nnx.merge(self.graphdefs, self.state)
        # Collect predictions and targets
        predictions_list, label_list = [], []
        for batch in tqdm(
            data.iter_batches(
                prefetch_batches=20,
                batch_size=batch_size,
                _collate_fn=functools.partial(pad_batch_spectra, num_peaks=num_peaks),
            ),
            desc="Evaluating",
        ):
            # Convert batch data to JAX arrays
            mz_array = jnp.array(batch["mz_array"])
            intensity_array = jnp.array(batch["intensity_array"])
            spectrum_mask = jnp.array(batch["spectrum_mask"])
            residue_indices = jnp.array(batch["residues_ids"])
            modification_indices = jnp.array(batch["modifications_ids"])
            peptide_mask = jnp.array(batch["peptide_mask"])
            label_list.append(jnp.array(batch["correct"]))

            preds = jax.nn.sigmoid(
                model(
                    mz_array=mz_array,
                    intensity_array=intensity_array,
                    spectrum_mask=spectrum_mask,
                    residue_indices=residue_indices,
                    modification_indices=modification_indices,
                    peptide_mask=peptide_mask,
                )
            )  # type: ignore[operator]
            predictions_list.append(preds)
        predictions = jnp.concatenate(predictions_list, axis=0)
        labels = jnp.concatenate(label_list, axis=0)

        # Evaluate and log metrics
        metrics = self.metrics_manager.evaluate(predictions, labels)
        self.logging_manager.log_metric(
            "eval/checkpoint_metric", metrics.metrics[metrics.checkpoint_metric], step
        )
        for metric_name, metric_value in metrics.metrics.items():
            if metric_name != metrics.checkpoint_metric:
                self.logging_manager.log_metric(
                    f"eval/{metric_name}", metric_value, step
                )
        return metrics.checkpoint_metric_value

    def __repr__(self) -> str:
        return self.__str__()
