from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx
from winnow.calibration.model.transformers import ConditionalTransformerEncoderBlock


# TODO: Double check positional encoding implementation
class PositionalEncoding(nnx.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        self.d_model = d_model
        self.max_len = max_len
        self.pe = self._create_positional_encoding()

    def _create_positional_encoding(self):
        position = jnp.arange(self.max_len)[:, jnp.newaxis]
        div_term = jnp.exp(
            jnp.arange(0, self.d_model, 2) * (-jnp.log(10000.0) / self.d_model)
        )
        pe = jnp.zeros((self.max_len, self.d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        return pe

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Add positional encoding to the input tensor."""
        return x + self.pe[jnp.newaxis, : x.shape[1], :]


class ModifiedResidueEncoder(nnx.Module):
    """Encodes modified residues into a fixed-size embedding."""

    def __init__(
        self,
        num_residues: int,
        num_modifications: int,
        dim_residue: int,
        dim_modification: int,
        dim_model: int,
        rngs: nnx.Rngs,
    ):
        """Initialize the ModifiedResidueEncoder.

        Args:
            dim_model: Dimension of the model (embedding size).
            rngs: Random number generators for initialization.
        """
        self.dim_model = dim_model
        self.dim_residue = dim_residue
        self.dim_modification = dim_modification
        self.num_residues = num_residues
        self.num_modifications = num_modifications

        # Initialize embeddings for residues and modifications
        self.residue_embedding = nnx.Embed(num_residues, dim_residue, rngs=rngs)
        self.modification_embedding = nnx.Embed(
            num_modifications, dim_modification, rngs=rngs
        )

        # Initialize projection layer
        self.projection = nnx.Linear(
            dim_residue + dim_modification, dim_model, rngs=rngs
        )

    def __call__(
        self, residue_indices: jnp.ndarray, modification_indices: jnp.ndarray
    ) -> jnp.ndarray:
        """Forward pass through the ModifiedResidueEncoder.

        Args:
            residue_indices: Array of residue indices of shape (batch_size, num_residues).

        Returns:
            Encoded residues of shape (batch_size, num_residues, dim_model).
        """
        residue_embeddings = self.residue_embedding(residue_indices)
        modification_embeddings = self.modification_embedding(modification_indices)
        # Concatenate residue and modification embeddings
        combined_embeddings = jnp.concatenate(
            [residue_embeddings, modification_embeddings], axis=-1
        )
        # Project to model dimension
        return self.projection(combined_embeddings)


@dataclass
class PeptideEncoderConfig:
    """Configuration for the ConditionalPeptideEncoder."""

    num_residues: int
    num_modifications: int
    dim_residue: int
    dim_modification: int
    dim_model: int
    num_self_heads: int
    num_condition_heads: int
    num_layers: int
    dim_feedforward: int
    dropout_rate: float
    max_len: int = 5000  # Default maximum length for positional encoding


class ConditionalPeptideEncoder(nnx.Module):
    """Encodes modified peptides using a transformer encoder with conditional attention."""

    def __init__(
        self,
        config: PeptideEncoderConfig,
        rngs: nnx.Rngs,
    ):
        self.config = config
        self.cls_embedding = nnx.Param(
            jax.random.normal(rngs.params(), (config.dim_model,), dtype=jnp.float32),
        )
        self.modified_residue_encoder = ModifiedResidueEncoder(
            num_residues=config.num_residues,
            num_modifications=config.num_modifications,
            dim_residue=config.dim_residue,
            dim_modification=config.dim_modification,
            dim_model=config.dim_model,
            rngs=rngs,
        )

        self.positional_encoding = PositionalEncoding(config.dim_model, config.max_len)
        self.transformer_blocks = [
            ConditionalTransformerEncoderBlock(
                dim_model=config.dim_model,
                num_self_heads=config.num_self_heads,
                num_condition_heads=config.num_condition_heads,
                dim_feedforward=config.dim_feedforward,
                dropout_rate=config.dropout_rate,
                rngs=rngs,
            )
            for _ in range(config.num_layers)
        ]

    def __call__(
        self,
        residue_ids: jnp.ndarray,
        modification_ids: jnp.ndarray,
        input_mask: jnp.ndarray,
        condition_embedding: jnp.ndarray,
        condition_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Forward pass through the encoder.

        Args:
            input_ids: Input token IDs.
            attention_mask: Attention mask to avoid attending to padding tokens.

        Returns:
            Output embeddings from the encoder.
        """
        # Add CLS token embedding
        cls_embeddings = jnp.repeat(
            self.cls_embedding[jnp.newaxis, jnp.newaxis, :],
            repeats=residue_ids.shape[0],
            axis=0,
        )
        residue_embeddings = self.modified_residue_encoder(
            residue_indices=residue_ids, modification_indices=modification_ids
        )
        embeddings = jnp.concatenate([cls_embeddings, residue_embeddings], axis=1)
        embeddings = self.positional_encoding(embeddings)

        # Add CLS token to the beginning of the sequence
        input_mask = jnp.concatenate(
            [
                jnp.full((input_mask.shape[0], 1), True, dtype=input_mask.dtype),
                input_mask,
            ],
            axis=1,
        )

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            embeddings = block(
                input_embedding=embeddings,
                input_mask=input_mask,
                condition_embedding=condition_embedding,
                condition_mask=condition_mask,
            )

        return embeddings[:, 0, :]  # Return only the CLS token embedding
