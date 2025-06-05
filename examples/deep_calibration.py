import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    from flax import nnx

    import optax

    import orbax.checkpoint as ocp

    import ray

    from sklearn.metrics import roc_auc_score

    from winnow.calibration.training.data import tokenizer

    from winnow.calibration.model.spectrum_encoder import SpectrumEncoderConfig
    from winnow.calibration.model.peptide_encoder import PeptideEncoderConfig
    from winnow.calibration.model.probability_calibrator import ProbabilityCalibrator, CalibratorConfig

    from winnow.calibration.training.loggers import LoggingManager, SystemLogger
    from winnow.calibration.training.trainer import Trainer, TrainerConfig, MetricsManager

    context = ray.init()
    print(context.dashboard_url)
    return (
        CalibratorConfig,
        LoggingManager,
        MetricsManager,
        PeptideEncoderConfig,
        ProbabilityCalibrator,
        SpectrumEncoderConfig,
        SystemLogger,
        Trainer,
        TrainerConfig,
        nnx,
        ocp,
        optax,
        ray,
        roc_auc_score,
        tokenizer,
    )


@app.cell
def _(ray):
    data_loader = ray.data.read_parquet('/Users/amandlamabona/Projects/winnow/input_data/processed/val.parquet', override_num_blocks=1200)
    data_loader
    return (data_loader,)


@app.cell
def _(
    CalibratorConfig,
    PeptideEncoderConfig,
    ProbabilityCalibrator,
    SpectrumEncoderConfig,
    nnx,
    tokenizer,
):
    calibrator_config = CalibratorConfig(
        spectrum_encoder=SpectrumEncoderConfig(
            dim_model=256, num_heads=8, dim_feedforward=512, num_layers=8, dropout_rate=0.0
        ),
        peptide_encoder=PeptideEncoderConfig(
            num_residues=len(tokenizer.residue_vocabulary),
            num_modifications=len(tokenizer.modifications_vocabulary),
            dim_residue=256, dim_modification=256, dim_model=256,
            num_self_heads=8, num_condition_heads=8, num_layers=8,
            dim_feedforward=512, dropout_rate=0.0, max_len=1000
        )
    )
    calibrator = ProbabilityCalibrator(config=calibrator_config, rngs=nnx.Rngs(42))
    calibrator
    return (calibrator,)


@app.cell
def _(calibrator, nnx, optax):
    optimizer = nnx.optimizer.Optimizer(
        model=calibrator, tx=optax.adam(learning_rate=1e-3)
    )
    return (optimizer,)


@app.cell
def _(LoggingManager, SystemLogger):
    logging_manager = LoggingManager(
        loggers=[SystemLogger(log_file='test_log.log')]
    )
    return (logging_manager,)


@app.cell
def _(MetricsManager, roc_auc_score):
    metrics_manager = MetricsManager(
        metrics={'auc': roc_auc_score}, checkpoint_metric='auc'
    )
    return (metrics_manager,)


@app.cell
def _(ocp):
    checkpoint_manager = ocp.CheckpointManager(directory='checkpoints/deep')
    return (checkpoint_manager,)


@app.cell
def _(TrainerConfig):
    train_config = TrainerConfig(
        num_epochs=1, batch_size=8, eval_batch_size=8, eval_frequency=20, patience=20, num_peaks=200
    )
    return (train_config,)


@app.cell
def _(
    Trainer,
    calibrator,
    checkpoint_manager,
    logging_manager,
    metrics_manager,
    optimizer,
):
    trainer = Trainer(
        model=calibrator, optimizer=optimizer, logging_manager=logging_manager,
        checkpoint_manager=checkpoint_manager, metrics_manager=metrics_manager
    )
    return (trainer,)


@app.cell
def _(data_loader):
    train_dataset = data_loader.random_sample(fraction=0.01)
    val_dataset = data_loader.random_sample(fraction=0.005)
    return train_dataset, val_dataset


@app.cell
def _(train_config, train_dataset, trainer, val_dataset):
    trainer.train(config=train_config, train_data=train_dataset, val_data=val_dataset)
    return


if __name__ == "__main__":
    app.run()
