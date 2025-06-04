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
        ConditionalPeptideEncoder,
        MODIFICATIONS,
        ModifiedPeptideTokenizer,
        PeptideEncoderConfig,
        ProbabilityCalibrator,
        RESIDUES,
        SpectrumEncoder,
        SpectrumEncoderConfig,
        functools,
        jnp,
        nnx,
        np,
        pl,
        ray,
        typing,
    )


@app.cell
def _(MODIFICATIONS, ModifiedPeptideTokenizer, RESIDUES):
    tokenizer = ModifiedPeptideTokenizer(residues=RESIDUES, modifications=MODIFICATIONS)
    return (tokenizer,)


@app.cell
def _(convert_to_str, functools, ray):
    spectrum_dataset = ray.data.read_parquet(
        '/Users/amandlamabona/Projects/winnow/input_data/spectrum_data/labelled/dataset-helaqc-annotated-0000-0001.parquet',
        columns=['global_index', 'mz_array', 'intensity_array'],
        override_num_blocks=1200
    )
    spectrum_dataset = spectrum_dataset.map(functools.partial(convert_to_str, column='mz_array')).map(functools.partial(convert_to_str, column='intensity_array'))
    spectrum_dataset
    return (spectrum_dataset,)


@app.cell
def _(np):
    def convert_to_str(row: dict, column: str) -> dict:
        row[column] = np.array(row[column]).tobytes()
        return row

    def convert_from_str(row: dict, column: str) -> dict:
        row[column] = np.frombuffer(row[column])
        return row
    return convert_from_str, convert_to_str


@app.cell
def _(spectrum_dataset):
    spectrum_dataset
    return


@app.cell
def _(pl):
    hela_predictions_data = pl.scan_csv('/Users/amandlamabona/Projects/winnow/input_data/beam_preds/labelled/helaqc-annotated_beam_preds.csv')
    hela_predictions_data = hela_predictions_data.select(
        pl.col('global_index'), pl.col('sequence').str.replace('L', 'I'), pl.col('preds').str.replace('L', 'I')
    )
    hela_predictions_data.collect().write_parquet('/Users/amandlamabona/Projects/winnow/input_data/beam_preds/labelled/helaqc-annotated_beam_preds.parquet')
    return (hela_predictions_data,)


@app.cell
def _(ray):
    predictions_dataset = ray.data.read_parquet(
        '/Users/amandlamabona/Projects/winnow/input_data/beam_preds/labelled/helaqc-annotated_beam_preds.parquet',
        override_num_blocks=1200
    )
    predictions_dataset
    return (predictions_dataset,)


@app.cell
def _(convert_from_str, functools, predictions_dataset, spectrum_dataset):
    full_dataset = spectrum_dataset.join(
        ds=predictions_dataset, join_type='inner', on=('global_index',), num_partitions=600
    )
    full_dataset = full_dataset.map(functools.partial(convert_from_str, column='mz_array')).map(functools.partial(convert_from_str, column='intensity_array'))
    full_dataset
    return


@app.cell
def _():
    #full_dataset.take(limit=5)
    return


@app.cell
def _(pl):
    hela_spectrum_data = pl.scan_parquet('/Users/amandlamabona/Projects/winnow/input_data/spectrum_data/labelled/dataset-helaqc-annotated-0000-0001.parquet')
    hela_spectrum_data = hela_spectrum_data.select(pl.col('global_index'), pl.col('mz_array'), pl.col('intensity_array'))
    return (hela_spectrum_data,)


@app.cell
def _(pl):
    spectrum_data = pl.scan_parquet('/Users/amandlamabona/Projects/winnow/input_data/spectrum_data/labelled/train_spectrum_all_datasets.parquet')
    spectrum_data.select(pl.col('global_index'), pl.col('sequence'))
    return


@app.cell
def _(hela_predictions_data, hela_spectrum_data, pl):
    hela_data = hela_spectrum_data.join(hela_predictions_data, on='global_index').with_columns(
        (pl.col('sequence') == pl.col('preds')).alias('correct')
    )
    hela_data.collect().write_parquet('/Users/amandlamabona/Projects/winnow/hela_data.parquet')
    return


@app.cell
def _(batch_spectra, functools, ray):
    data_loader = ray.data.read_parquet('/Users/amandlamabona/Projects/winnow/hela_data.parquet', override_num_blocks=1200)
    data_loader = data_loader.map_batches(functools.partial(batch_spectra, num_peaks=200))
    return (data_loader,)


@app.cell
def _(jnp, typing):
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
        batch['padding_mask'] = (jnp.arange(0, num_peaks).reshape(1, -1) < length_array.reshape(-1, 1)).astype(int)
        return batch
    return (batch_spectra,)


@app.cell
def _(data_loader):
    batch = data_loader.take_batch(batch_size=8)
    return (batch,)


@app.cell
def _(SpectrumEncoder, SpectrumEncoderConfig, nnx):
    spectrum_encoder_config = SpectrumEncoderConfig(
        dim_model=256, num_heads=16, dim_feedforward=512, num_layers=8, dropout_rate=0.0
    )
    spectrum_encoder = SpectrumEncoder(config=spectrum_encoder_config,rngs=nnx.Rngs(42))
    spectrum_encoder
    return (spectrum_encoder,)


@app.cell
def _(batch):
    batch['padding_mask'].shape
    return


@app.cell
def _(batch, spectrum_encoder):
    spectrum_embedding = spectrum_encoder(mz_array=batch['mz_array'], intensity_array=batch['intensity_array'], mask=batch['padding_mask'])
    return (spectrum_embedding,)


@app.cell
def _(batch, jnp):
    batch['padding_mask'][:, jnp.newaxis, :, jnp.newaxis].ndim
    return


@app.cell
def _(batch, tokenizer):
    tokenizer.detokenize(*tokenizer.tokenize(peptide=batch['sequence'][-2])) == batch['sequence'][-2]
    return


@app.cell
def _(tokenizer):
    tokenizer.residue_vocabulary[tokenizer.residue_pad_index]
    return


@app.cell
def _(tokenizer):
    tokenizer.modifications_vocabulary[tokenizer.modification_pad_index]
    return


@app.cell
def _(batch, tokenizer):
    residues_ids, modifications_ids, peptide_mask = tokenizer.pad_tokenize(peptides=batch['sequence'])
    return modifications_ids, peptide_mask, residues_ids


@app.cell
def _(modifications_ids, peptide_mask, residues_ids):
    residues_ids.shape, modifications_ids.shape, peptide_mask.shape
    return


@app.cell
def _(ConditionalPeptideEncoder, PeptideEncoderConfig, nnx, tokenizer):
    peptide_encoder_config = PeptideEncoderConfig(
        num_residues=len(tokenizer.residue_vocabulary),
        num_modifications=len(tokenizer.modifications_vocabulary),
        dim_residue=256, dim_modification=256, dim_feedforward=512,
        dim_model=256, dropout_rate=0.0, num_self_heads=16,
        num_condition_heads=16, num_layers=8, max_len=1000
    )
    peptide_encoder = ConditionalPeptideEncoder(
        config=peptide_encoder_config, rngs=nnx.Rngs(42)
    )
    return (peptide_encoder,)


@app.cell
def _(spectrum_embedding):
    spectrum_embedding.shape
    return


@app.cell
def _(jnp, peptide_mask):
    padded_peptide_mask = jnp.concatenate([jnp.full(fill_value=True, shape=(8, 1, 1, 1)), peptide_mask[:, jnp.newaxis, :, jnp.newaxis]], axis=-2)
    return


@app.cell
def _(
    batch,
    modifications_ids,
    peptide_encoder,
    peptide_mask,
    residues_ids,
    spectrum_embedding,
):
    peptide_encoder(
        residue_ids=residues_ids,
        modification_ids=modifications_ids,
        input_mask=peptide_mask, #peptide_mask[:, jnp.newaxis, :, jnp.newaxis],
        condition_embedding=spectrum_embedding,
        condition_mask=batch['padding_mask']
    ).shape
    return


@app.cell
def _(batch, jnp):
    batch['padding_mask'][:, jnp.newaxis, :, jnp.newaxis].shape
    return


@app.cell
def _(peptide_encoder):
    peptide_encoder
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
def _(batch, calibrator, modifications_ids, peptide_mask, residues_ids):
    calibrator(
        mz_array=batch['mz_array'],
        intensity_array=batch['intensity_array'],
        spectrum_mask=batch['padding_mask'],
        residue_indices=residues_ids,
        modification_indices=modifications_ids,
        peptide_mask=peptide_mask
    )
    return


@app.cell
def _(batch):
    batch
    return


@app.cell
def _(batch, jnp):
    jnp.array(batch['correct'], dtype=int)
    return


@app.cell
def _(data_loader):
    batch_counter = 0
    for local_batch in data_loader.iter_batches(batch_size=8):
        print('Batch received', batch_counter)
        batch_counter += 1
    return


if __name__ == "__main__":
    app.run()
