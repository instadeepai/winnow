import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo


    import ast
    import functools
    import pickle
    import time
    import typing

    import numpy as np

    import jax
    import jax.numpy as jnp

    from flax import nnx

    import ray
    import ray.data

    import polars as pl

    from winnow.calibration.training.loggers import NeptuneLogger

    from winnow.calibration.model.spectrum_encoder import SpectrumEncoderConfig, SpectrumEncoder
    from winnow.calibration.model.peptide_encoder import PeptideEncoderConfig, ConditionalPeptideEncoder
    from winnow.calibration.model.probability_calibrator import ProbabilityCalibrator, CalibratorConfig

    from winnow.constants import RESIDUES, MODIFICATIONS
    from winnow.calibration.model.tokenizer import ModifiedPeptideTokenizer

    context = ray.init()
    print(context.dashboard_url)
    return (
        CalibratorConfig,
        MODIFICATIONS,
        ModifiedPeptideTokenizer,
        PeptideEncoderConfig,
        ProbabilityCalibrator,
        RESIDUES,
        SpectrumEncoderConfig,
        functools,
        jnp,
        nnx,
        pl,
        ray,
        typing,
    )


@app.cell
def _(MODIFICATIONS, ModifiedPeptideTokenizer, RESIDUES):
    tokenizer = ModifiedPeptideTokenizer(residues=RESIDUES, modifications=MODIFICATIONS)
    return (tokenizer,)


@app.cell
def _(pl):
    SPECIES='immuno'
    predictions_data = pl.scan_csv(f'/Users/amandlamabona/Projects/winnow/input_data/beam_preds/labelled/{SPECIES}-annotated_beam_preds.csv')
    predictions_data = predictions_data.select(
        pl.col('global_index'), pl.col('sequence').str.replace('L', 'I'), pl.col('preds').str.replace('L', 'I')
    ).filter(
        pl.col('preds').is_not_null() & pl.col('sequence').is_not_null()
    )

    spectrum_data = pl.scan_parquet(f'/Users/amandlamabona/Projects/winnow/input_data/spectrum_data/labelled/dataset-{SPECIES}-annotated-0000-0001.parquet')
    spectrum_data = spectrum_data.select(pl.col('global_index'), pl.col('mz_array'), pl.col('intensity_array'))

    data = spectrum_data.join(predictions_data, on='global_index').with_columns(
        (pl.col('sequence') == pl.col('preds')).cast(pl.Int8).alias('correct')
    )
    data.collect().write_parquet(f'/Users/amandlamabona/Projects/winnow/input_data/processed/{SPECIES}.parquet')
    return


@app.cell
def _(batch_spectra, functools, ray):
    data_loader = ray.data.read_parquet('/Users/amandlamabona/Projects/winnow/input_data/processed/val.parquet', override_num_blocks=1200)
    data_loader = data_loader.filter(expr='preds is not None')
    data_loader = data_loader.map_batches(functools.partial(batch_spectra, num_peaks=200))
    return (data_loader,)


@app.cell
def _(jnp, tokenizer, typing):
    def batch_spectra(
        batch: typing.Dict[str, jnp.ndarray], num_peaks: int
    ) -> typing.Dict[str, jnp.ndarray]:
        # Initialize batch variables
        mz_list, intensities_list, length_list = [], [], []

        # Pad batch
        for mz_array, intensities_array in list(zip(batch['mz_array'], batch['intensity_array'])):
            # Coerce spectra into jax arrays
            local_mzs = jnp.array(mz_array)
            local_intensities = jnp.array(intensities_array)

            # Extract peaks with highest intensities
            length = local_mzs.shape[0]
            if length < num_peaks:
                local_mzs = jnp.pad(local_mzs, (0, num_peaks - length))
                local_intensities = jnp.pad(local_intensities, (0, num_peaks - length))
            else:
                local_sort_idxs = jnp.argpartition(local_intensities, -num_peaks)
                local_mzs, local_intensities = local_mzs[local_sort_idxs[-num_peaks:]], local_intensities[local_sort_idxs[-num_peaks:]]

            # Pad spectra
            assert local_mzs.shape[0] == local_intensities.shape[0]

            # Collate spectra and lengths for masking
            mz_list.append(local_mzs)
            intensities_list.append(local_intensities)
            length_list.append(length)

        # Batch padded spectra
        batch['mz_array'] = jnp.stack(mz_list)
        batch['intensity_array'] = jnp.stack(intensities_list)

        # Generate padding mask
        length_array = jnp.array(length_list)
        batch['spectrum_mask'] = jnp.arange(0, num_peaks).reshape(1, -1) < length_array.reshape(-1, 1)

        # Coerce labels
        batch['correct'] = jnp.array(batch['correct'], dtype=int)

        # Batch peptides
        batch['residues_ids'], batch['modifications_ids'], batch['peptide_mask'] = tokenizer.pad_tokenize(peptides=batch['sequence'])
        return batch
    return (batch_spectra,)


@app.cell
def _(data_loader):
    batch = data_loader.take_batch(batch_size=8)
    return (batch,)


@app.cell
def _(batch):
    batch
    return


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
    return (calibrator,)


@app.cell
def _(calibrator):
    calibrator
    return


@app.cell
def _(batch, calibrator):
    calibrator(
        mz_array=batch['mz_array'],
        intensity_array=batch['intensity_array'],
        spectrum_mask=batch['spectrum_mask'],
        residue_indices=batch['residues_ids'],
        modification_indices=batch['modifications_ids'],
        peptide_mask=batch['peptide_mask']
    )
    return


@app.cell
def _(data_loader):
    batch_counter = 0
    for local_batch in data_loader.iter_batches(batch_size=8):
        print('Batch received!', batch_counter)
        batch_counter += 1
    return


if __name__ == "__main__":
    app.run()
