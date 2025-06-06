import os
import hydra
from omegaconf import DictConfig

import jax

from flax import nnx

import optax

import orbax.checkpoint as ocp

import ray

from pyarrow.fs import S3FileSystem

from sklearn.metrics import roc_auc_score

from winnow.calibration.training.data import tokenizer

from winnow.calibration.model.spectrum_encoder import SpectrumEncoderConfig
from winnow.calibration.model.peptide_encoder import PeptideEncoderConfig
from winnow.calibration.model.probability_calibrator import ProbabilityCalibrator, CalibratorConfig

from winnow.calibration.training.loggers import LoggingManager, SystemLogger
from winnow.calibration.training.trainer import Trainer, TrainerConfig, MetricsManager

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_platform_name", 'gpu')

@hydra.main(version_base=None, config_path='../../configs', config_name='deep_calibration')
def main(config: DictConfig):
    """ Main function to run the deep calibration training. """
    # Initialize Ray for distributed data processing 
    ray.init()

    # Load the dataset from Parquet files   
    train_dataset = ray.data.read_parquet(
        os.path.join(os.environ["AICHOR_INPUT_PATH"], config.data.train_path),
        filesystem=S3FileSystem(endpoint_override=os.environ["AWS_ENDPOINT_URL"]),
        override_num_blocks=1200
    )
    val_dataset = ray.data.read_parquet(
        os.path.join(os.environ["AICHOR_INPUT_PATH"], config.data.val_path),
        filesystem=S3FileSystem(endpoint_override=os.environ["AWS_ENDPOINT_URL"]),
        override_num_blocks=1200
    )

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
        loggers=[SystemLogger(log_file=config.logging.log_file)],
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

if __name__ == '__main__':
    main()
