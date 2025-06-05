from dataclasses import dataclass

import jax.numpy as jnp
from flax import nnx

from winnow.calibration.model.spectrum_encoder import (
    SpectrumEncoder,
    SpectrumEncoderConfig,
)
from winnow.calibration.model.peptide_encoder import (
    ConditionalPeptideEncoder,
    PeptideEncoderConfig,
)


@dataclass
class CalibratorConfig:
    """Configuration for the probability calibrator."""

    spectrum_encoder: SpectrumEncoderConfig
    peptide_encoder: PeptideEncoderConfig


class ProbabilityCalibrator(nnx.Module):
    """Probability calibrator for peptide-spectrum matches."""

    def __init__(self, config: CalibratorConfig, rngs: nnx.Rngs):
        """Initialize the ProbabilityCalibrator.

        Args:
            config: Configuration for the calibrator, including spectrum and peptide encoders.
            rngs: Random number generators for initialization.
        """
        self.config = config
        self.spectrum_encoder = SpectrumEncoder(config.spectrum_encoder, rngs=rngs)
        self.peptide_encoder = ConditionalPeptideEncoder(
            config.peptide_encoder, rngs=rngs
        )
        self.head = nnx.Linear(config.peptide_encoder.dim_model, 1, rngs=rngs)

    def __call__(
        self,
        mz_array: jnp.ndarray,
        intensity_array: jnp.ndarray,
        spectrum_mask: jnp.ndarray,
        residue_indices: jnp.ndarray,
        modification_indices: jnp.ndarray,
        peptide_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Forward pass through the probability calibrator.

        Args:
            mz_array: Mass spectra of shape (batch_size, num_spectra, spectrum_dim).
            intensity_array: Intensity values of shape (batch_size, num_spectra, spectrum_dim).
            spectrum_mask: Mask for the spectra of shape (batch_size, num_spectra).
            residue_indices: Indices of the residues in the peptides.
            modification_indices: Indices of the modifications in the peptides.
            peptide_mask: Mask for the peptides.

        Returns:
            Calibrated probabilities of shape (batch_size, num_peptides).
        """
        encoded_spectra = self.spectrum_encoder(
            mz_array=mz_array, intensity_array=intensity_array, mask=spectrum_mask
        )
        encoded_peptides = self.peptide_encoder(
            residue_ids=residue_indices,
            modification_ids=modification_indices,
            input_mask=peptide_mask,
            condition_embedding=encoded_spectra,
            condition_mask=spectrum_mask,
        )

        # Apply a final linear layer to get probabilities
        logits = self.head(encoded_peptides)

        return jnp.squeeze(logits, -1)  # Ensure probabilities are in [0, 1] range
