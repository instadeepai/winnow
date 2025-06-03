import jax.numpy as jnp
from flax import nnx

class TransformerEncoderBlock(nnx.Module):
    """Transformer encoder block with self-attention and feedforward layers."""
    def __init__(
        self, dim_model: int, num_heads: int, dim_feedforward: int, dropout_rate: float,
        rngs: nnx.Rngs
    ):
        """Initialize the Transformer encoder block.
        Args:
            dim_model: Dimension of the model (embedding size).
            num_heads: Number of attention heads.
            dim_feedforward: Dimension of the feedforward network.
            dropout_rate: Dropout rate for regularization.
            rngs: Random number generators for initialization.
        """
        # Self attention block
        self.self_attn = nnx.MultiHeadAttention(
            num_heads=num_heads, in_features=dim_model,
            qkv_features=dim_model,
            dropout_rate=dropout_rate,
            decode=False,
            rngs=rngs
        )
        self.self_attn_layernorm = nnx.LayerNorm(num_features=dim_model, rngs=rngs)
        
        # Feedforward block
        self.feedforward_model = nnx.Sequential(
            nnx.Linear(dim_model, dim_feedforward, rngs=rngs),
            nnx.relu, nnx.Dropout(rate=dropout_rate, rngs=rngs),
            nnx.Linear(dim_feedforward, dim_model, rngs=rngs)
        )
        self.feedforward_layernorm = nnx.LayerNorm(num_features=dim_model, rngs=rngs)

    def __call__(self, input_embedding: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        """Forward pass through the Transformer encoder block.
        Args:
            input_embedding: Input embeddings of shape (batch_size, seq_len, dim_model).
            mask: Attention mask of shape (batch_size, seq_len, seq_len).
        Returns:
            Output embeddings of shape (batch_size, seq_len, dim_model).
        """
        if mask.ndim == 2:
            mask = mask[:, jnp.newaxis, :, jnp.newaxis]
        intermediate_embedding = self.self_attn_layernorm(
            input_embedding + self.self_attn(input_embedding, mask=mask)
        )
        output_embedding = self.feedforward_layernorm(
            intermediate_embedding + self.feedforward_model(intermediate_embedding)
        )
        return output_embedding


class ConditionalTransformerEncoderBlock(nnx.Module):
    def __init__(
        self, dim_model: int, num_self_heads: int, dim_feedforward: int, 
        num_condition_heads: int, dropout_rate: float,
        rngs: nnx.Rngs
    ):
        """Initialize the Conditional Transformer encoder block.
        Args:
            dim_model: Dimension of the model (embedding size).
            num_heads: Number of attention heads.
            dim_feedforward: Dimension of the feedforward network.
            dim_condition: Dimension of the conditioning vector.
            dropout_rate: Dropout rate for regularization.
            rngs: Random number generators for initialization.
        """
        # Self attention block
        self.self_attn = nnx.MultiHeadAttention(
            num_heads=num_self_heads,
            in_features=dim_model,
            qkv_features=dim_model,
            dropout_rate=dropout_rate,
            decode=False,
            rngs=rngs
        )
        self.self_attn_layernorm = nnx.LayerNorm(num_features=dim_model, rngs=rngs)
        
        # Conditional attention block
        self.conditional_attn = nnx.MultiHeadAttention(
            num_heads=num_condition_heads,
            in_features=dim_model,
            qkv_features=dim_model,
            dropout_rate=dropout_rate,
            decode=False,
            rngs=rngs
        )
        self.conditional_attn_layernorm = nnx.LayerNorm(num_features=dim_model, rngs=rngs)
        
        # Feedforward block
        self.feedforward_model = nnx.Sequential(
            nnx.Linear(dim_model, dim_feedforward, rngs=rngs),
            nnx.relu, nnx.Dropout(rate=dropout_rate, rngs=rngs),
            nnx.Linear(dim_feedforward, dim_model, rngs=rngs)
        )
        self.feedforward_layernorm = nnx.LayerNorm(num_features=dim_model, rngs=rngs)

    def __call__(
        self, input_embedding: jnp.ndarray, condition_embedding: jnp.ndarray, input_mask: jnp.ndarray,
        condition_mask: jnp.ndarray
    ) -> jnp.ndarray:
        """Forward pass through the Conditional Transformer encoder block.
        Args:
            input_embedding: Input embeddings of shape (batch_size, seq_len, dim_model).
            condition_embedding: Conditioning embeddings of shape (batch_size, cond_seq_len, dim_condition).
            input_mask: Attention mask for input of shape (batch_size, seq_len, seq_len).
            condition_mask: Attention mask for condition of shape (batch_size, cond_seq_len, cond_seq_len).
        Returns:
            Output embeddings of shape (batch_size, seq_len, dim_model).
        """
        if input_mask.ndim == 2:
            input_mask = input_mask[:, jnp.newaxis, jnp.newaxis, :] # originally (batch_size, seq_len), now (batch_size, num_self_heads, seq_len, seq_len)
        if condition_mask.ndim == 2:
            condition_mask = condition_mask[:, jnp.newaxis, jnp.newaxis, :] # originally (batch_size, cond_seq_len), now (batch_size, num_condition_heads, seq_len, cond_seq_len)
        
        intermediate_embedding = self.self_attn_layernorm(
            input_embedding + self.self_attn(input_embedding, mask=input_mask)
        )
        intermediate_embedding = self.conditional_attn_layernorm(
            intermediate_embedding + self.conditional_attn(
                intermediate_embedding, condition_embedding, mask=condition_mask
            )
        )
        output_embedding = self.feedforward_layernorm(
            intermediate_embedding + self.feedforward_model(intermediate_embedding)
        )
        return output_embedding