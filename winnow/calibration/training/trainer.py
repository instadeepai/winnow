"""
Interfaces for the Trainer.
"""
import dataclasses
from typing import Callable, Dict

import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx
from jaxtyping import Float
from ray.data import Dataset
from tqdm import tqdm

from .loggers import LoggingManager


@dataclasses.dataclass
class Metrics:
    """
    Metrics for the model.
    """
    metrics: Dict[str, float]
    checkpoint_metric: str
    checkpoint_metric_value: float


@dataclasses.dataclass
class MetricsManager:
    """
    Manager that logs metrics to the console and Neptune.
    """
    metrics: Dict[
        str,
        Callable[
            [Float[jnp.ndarray, " batch_size ..."],
             Float[jnp.ndarray, " batch_size ..."]],
            float]
    ]
    checkpoint_metric: str

    def evaluate(
        self,
        y_true: Float[jnp.ndarray, " batch_size ..."],
        y_pred: Float[jnp.ndarray, " batch_size ..."]
    ) -> Metrics:
        """
        Evaluate the model.
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
            checkpoint_metric_value=checkpoint_metric
        )


@dataclasses.dataclass
class TrainerConfig:
    """
    Configuration for the trainer.
    """
    input_column: str
    label_column: str
    num_epochs: int
    batch_size: int
    eval_frequency: int
    eval_batch_size: int
    learning_rate: float
    patience: int


class Trainer:
    """
    Trainer that trains a model.
    """
    def __init__(
        self,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        loss_fn: Callable[
            [Float[jnp.ndarray, " batch_size ..."],
             Float[jnp.ndarray, " batch_size ..."]],
            Float[jnp.ndarray, " batch_size ..."]
        ],
        logging_manager: LoggingManager,
        checkpoint_manager: ocp.CheckpointManager,
        metrics_manager: MetricsManager
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.logging_manager = logging_manager
        self.checkpoint_manager = checkpoint_manager
        self.metrics_manager = metrics_manager

        def wrapped_loss_fn(
            model: nnx.Module,
            xs: Float[jnp.ndarray, " batch_size ..."],
            ys: Float[jnp.ndarray, " batch_size ..."]
        ) -> Float[jnp.ndarray, " batch_size ..."]:
            """
            Loss function.
            """
            y_preds = model(xs) # type: ignore[operator]
            return self.loss_fn(y_preds, ys)

        def update(
            model: nnx.Module, optimizer: nnx.Optimizer,
            xs: Float[jnp.ndarray, " batch_size ..."],
            ys: Float[jnp.ndarray, " batch_size ..."]
        ) -> float:
            """
            Update the model.
            """
            loss, grads = nnx.value_and_grad(wrapped_loss_fn)(model, xs, ys)
            optimizer.update(grads)
            return loss

        self.update = nnx.jit(update)

    def train(
        self, train_data: Dataset, val_data: Dataset, config: TrainerConfig
    ) -> None:
        """
        Train the model.
        Args:
            train_data: The dataset to train on.
            val_data: The dataset to validate on.
            config: The configuration for the trainer.
        """
        step, evals_without_improvement, best_checkpoint_metric = 0, 0, float("-inf")
        for _ in tqdm(range(config.num_epochs), desc="Epochs"):
            for batch in tqdm(
                train_data.iter_batches(batch_size=config.batch_size), desc="Batches"
            ):
                xs = jnp.array(batch[config.input_column])
                ys = jnp.array(batch[config.label_column])
                loss = self.update(self.model, self.optimizer, xs, ys)
                self.logging_manager.log_metric("train/loss", loss, step)
                step += 1
                if step % config.eval_frequency == 0:
                    checkpoint_metric = self.evaluate(
                        val_data, config.eval_batch_size, step
                    )
                    if checkpoint_metric > best_checkpoint_metric:
                        evals_without_improvement = 0
                        best_checkpoint_metric = checkpoint_metric
                        _, state = nnx.split(self.model)
                        self.checkpoint_manager.save(
                            step, args=ocp.args.StandardSave(state)
                        )
                    else:
                        evals_without_improvement += 1
                    if evals_without_improvement >= config.patience:
                        break

    def evaluate(self, data: Dataset, batch_size: int, step: int) -> float:
        """
        Evaluate the model.
        Args:
            data: The dataset to predict on.
            batch_size: The batch size.
            step: The current step.
        """
        # Collect predictions and targets
        predictions_list, ys_list = [], []
        for batch in data.iter_batches(batch_size=batch_size):
            xs = jnp.array(batch["x"])
            ys_list.append(jnp.array(batch["y"]))
            preds = self.model(xs) # type: ignore[operator]
            predictions_list.append(preds)
        predictions = jnp.concatenate(predictions_list, axis=0)
        ys = jnp.concatenate(ys_list, axis=0)

        # Evaluate and log metrics
        metrics = self.metrics_manager.evaluate(ys, predictions)
        self.logging_manager.log_metric(
            "eval/checkpoint_metric", metrics.metrics[metrics.checkpoint_metric], step
        )
        for metric_name, metric_value in metrics.metrics.items():
            if metric_name != metrics.checkpoint_metric:
                self.logging_manager.log_metric(
                    f"eval/{metric_name}", metric_value, step
                )
        return metrics.checkpoint_metric_value

    def __str__(self) -> str:
        return f"""
Trainer(
    model={self.model},
    optimizer={self.optimizer},
    loss_fn={self.loss_fn},
    logging_manager={self.logging_manager},
    checkpoint_manager={self.checkpoint_manager},
    metrics_manager={self.metrics_manager}
)
        """

    def __repr__(self) -> str:
        return self.__str__()
