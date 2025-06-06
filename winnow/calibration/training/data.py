import typing

import jax.numpy as jnp

from winnow.constants import MODIFICATIONS, RESIDUES
from winnow.calibration.model.tokenizer import ModifiedPeptideTokenizer

tokenizer = ModifiedPeptideTokenizer(
    residues=RESIDUES,
    modifications=MODIFICATIONS,
)

def pad_batch_spectra(
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

    # Batch peptides
    batch['residues_ids'], batch['modifications_ids'], batch['peptide_mask'] = tokenizer.pad_tokenize(peptides=batch['sequence'])
    return batch