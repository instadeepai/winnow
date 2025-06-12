import functools
import os
import hydra
from omegaconf import DictConfig

import jax

from flax import nnx

import optax

import orbax.checkpoint as ocp

import ray
import polars as pl

from sklearn.metrics import roc_auc_score

from winnow.calibration.training.data import (
    tokenizer#, RayActorDataLoader, pad_batch_spectra
)

from winnow.calibration.model.spectrum_encoder import SpectrumEncoderConfig
from winnow.calibration.model.peptide_encoder import PeptideEncoderConfig
from winnow.calibration.model.probability_calibrator import ProbabilityCalibrator, CalibratorConfig

from winnow.calibration.training.loggers import LoggingManager, SystemLogger, NeptuneLogger
from winnow.calibration.training.trainer import Trainer, TrainerConfig, MetricsManager

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_platform_name", 'gpu')

@hydra.main(version_base=None, config_path='../../configs', config_name='deep_calibration')
def main(config: DictConfig):
    """ Main function to run the deep calibration training. """
    # Initialize Ray for distributed data processing
    ray.init()

    # Load the dataset from Parquet files   
    train_df = pl.read_parquet(
            os.path.join(config.data.bucket_path, config.data.train_path),
            storage_options={
                "aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
                "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
                "aws_endpoint_url": os.environ["AWS_ENDPOINT_URL"],
            }
        )
    print("Length of train_df:", len(train_df))
    train_df = train_df.with_columns(
            pl.col('preds').map_elements(
                lambda sequence: len(tokenizer.tokenize(sequence)[0])
            ).alias('preds_length')
        ).filter(pl.col('preds_length') < config.training.max_length)
    print("Length of train_df after filtering:", len(train_df))
    train_dataset = ray.data.from_arrow(train_df.to_arrow())

    val_df = pl.read_parquet(
            os.path.join(config.data.bucket_path, config.data.val_path),
            storage_options={
                "aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
                "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
                "aws_endpoint_url": os.environ["AWS_ENDPOINT_URL"],
            }
    )
    print("Length of val_df:", len(val_df))
    val_df = val_df.with_columns(
            pl.col('preds').map_elements(
                lambda sequence: len(tokenizer.tokenize(sequence)[0])
            ).alias('preds_length')
        ).filter(pl.col('preds_length') < config.training.max_length)
    print("Length of val_df after filtering:", len(val_df))
    val_dataset = ray.data.from_arrow(val_df.to_arrow())

    # Initialize model
    calibrator_config = CalibratorConfig(
        spectrum_encoder=SpectrumEncoderConfig(**config.model.spectrum_encoder),
        peptide_encoder=PeptideEncoderConfig(
            num_residues=len(tokenizer.residue_vocabulary),
            num_modifications=len(tokenizer.modifications_vocabulary),
            **config.model.peptide_encoder
        )
    )
    calibrator = ProbabilityCalibrator(config=calibrator_config, rngs=nnx.Rngs(config.model.seed))

    # Initialize the optimizer and training
    optimizer = nnx.optimizer.Optimizer(
        model=calibrator, tx=optax.adam(learning_rate=config.training.learning_rate),
    )

    # Initialize logging, metrics, and checkpoint managers
    logging_manager = LoggingManager(
        loggers=[
            SystemLogger(log_file=config.logging.log_file),
            NeptuneLogger(project_name=config.logging.neptune_project,
                          api_token=os.environ["NEPTUNE_API_TOKEN"])
        ],
    )

    metrics_manager = MetricsManager(
        metrics={'auc': roc_auc_score}, checkpoint_metric='auc'
    )

    checkpoint_manager = ocp.CheckpointManager(directory=config.checkpointing.checkpoint_path)

    train_config = TrainerConfig(**config.training)

    # Create the trainer and start training
    trainer = Trainer(
        model=calibrator, optimizer=optimizer, logging_manager=logging_manager,
        checkpoint_manager=checkpoint_manager, metrics_manager=metrics_manager
    )
    trainer.train(config=train_config, train_data=train_dataset, val_data=val_dataset)
    if config.logging.ray_timeline_path:
        ray.timeline(filename=config.logging.ray_timeline_path)

if __name__ == '__main__':
    main()
