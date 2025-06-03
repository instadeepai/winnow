from dataclasses import dataclass

import jax.numpy as jnp
from flax import nnx
from winnow.calibration.model.transformers import TransformerEncoderBlock

class PeakEncoder(nnx.Module):
    """Encodes peaks from a spectrum into a fixed-size embedding."""
    def __init__(self, dim_model: int, rngs: nnx.Rngs):
        """Initialize the PeakEncoder.
        Args:
            dim_model: Dimension of the model (embedding size).
            rngs: Random number generators for initialization.
        """
        self.dim_model = dim_model
        self.projection = nnx.Linear(2, dim_model, rngs=rngs)

    def __call__(self, mz_array: jnp.ndarray, intensity_array: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the PeakEncoder.
        Args:
            mz_array: Array of m/z values of shape (batch_size, num_peaks).
            intensity_array: Array of intensity values of shape (batch_size, num_peaks).
        Returns:
            Encoded peaks of shape (batch_size, num_peaks, dim_model).
        """
        return self.projection(jnp.stack([mz_array, intensity_array], -1))

@dataclass
class SpectrumEncoderConfig:
    """Configuration for the SpectrumEncoder."""
    dim_model: int = 128
    num_heads: int = 8
    dim_feedforward: int = 512
    num_layers: int = 6
    dropout_rate: float = 0.1

class SpectrumEncoder(nnx.Module):
    """Encodes a spectrum using a transformer encoder."""
    def __init__(
        self, config: SpectrumEncoderConfig, rngs: nnx.Rngs
    ):
        """Initialize the SpectrumEncoder.
        Args:
            dim_model: Dimension of the model (embedding size).
            num_heads: Number of attention heads.
            dim_feedforward: Dimension of the feedforward network.
            num_layers: Number of transformer encoder layers.
            dropout_rate: Dropout rate for regularization.
            rngs: Random number generators for initialization.
        """
        self.config = config
        self.rngs = rngs
        self.peak_encoder = PeakEncoder(dim_model=config.dim_model, rngs=rngs)
        self.encoder_layers = [
            TransformerEncoderBlock(
                dim_model=config.dim_model, num_heads=config.num_heads, dim_feedforward=config.dim_feedforward,
                dropout_rate=config.dropout_rate, rngs=rngs
            ) for _ in range(config.num_layers)
        ]
    
    def __call__(
        self, mz_array: jnp.ndarray, intensity_array: jnp.ndarray, mask: jnp.ndarray
    ) -> jnp.ndarray:
        """Forward pass through the SpectrumEncoder.
        Args:
            mz_array: Array of m/z values of shape (batch_size, num_peaks).
            intensity_array: Array of intensity values of shape (batch_size, num_peaks).
            mask: Attention mask of shape (batch_size, num_peaks, num_peaks).
        Returns:
            Encoded spectrum of shape (batch_size, num_peaks, dim_model).
        """
        embeddings = self.peak_encoder(mz_array=mz_array, intensity_array=intensity_array)
        for layer in self.encoder_layers:
            embeddings = layer(input_embedding=embeddings, mask=mask)
        return embeddings