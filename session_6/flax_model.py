"""
flax_model.py
-------------
Flax/Linen implementation of the Transformer encoder block for Session 5.

Architecture mirrors BenchmarkTransformerBlock from transformer_block.py:
  - Post-LayerNorm multi-head self-attention
  - Position-wise feedforward: Dense → GELU → Dense
  - Residual connections around both sub-layers

This file is Session 5 only — not imported by any other session.
Import from a Session 5 notebook with:

    import sys; sys.path.insert(0, ".")
    from flax_model import FlaxTransformerBlock, D_MODEL, N_HEAD, DIM_FEEDFORWARD, SEQ_LEN
"""

import flax.linen as nn
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Hyperparameters — match transformer_block.py constants exactly
# ---------------------------------------------------------------------------
D_MODEL         = 512
N_HEAD          = 8
DIM_FEEDFORWARD = 2048
SEQ_LEN         = 128


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class FlaxTransformerBlock(nn.Module):
    """
    Single Transformer encoder block in Flax/Linen.

    Matches the architecture of BenchmarkTransformerBlock exactly:
        x = LayerNorm(x + MHA(x, x, x))
        x = LayerNorm(x + FFN(x))

    Parameters are managed as a pytree via model.init() and passed
    explicitly to model.apply() — there is no stateful model object.
    """
    d_model:         int = D_MODEL
    nhead:           int = N_HEAD
    dim_feedforward: int = DIM_FEEDFORWARD

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # --- Multi-head self-attention + residual + layer norm ---
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.nhead,
            qkv_features=self.d_model,
            out_features=self.d_model,
        )(x, x)
        x = nn.LayerNorm()(x + attn_out)

        # --- Position-wise feedforward + residual + layer norm ---
        ff = nn.Dense(self.dim_feedforward)(x)
        ff = nn.gelu(ff)
        ff = nn.Dense(self.d_model)(ff)
        x = nn.LayerNorm()(x + ff)

        return x
