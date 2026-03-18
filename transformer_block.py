"""
transformer_block.py
--------------------
Shared model definition for the GPU/TPU workshop benchmark sessions.

Import this module from any session notebook:

    import sys; sys.path.insert(0, "..")
    from transformer_block import BenchmarkTransformerBlock, D_MODEL, N_HEAD, DIM_FEEDFORWARD, SEQ_LEN

Each session owns its own benchmark loop, timing logic, and result serialization.
"""

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Default hyperparameters — consistent across all sessions
# ---------------------------------------------------------------------------
D_MODEL         = 512
N_HEAD          = 8
DIM_FEEDFORWARD = 2048
SEQ_LEN         = 128   # tokens per sequence (sessions may override locally)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class BenchmarkTransformerBlock(nn.Module):
    """
    A single Transformer encoder block: multi-head self-attention + feed-forward.
    No dynamic control flow — fully static graph, compatible with all runtimes
    (CPU, GPU via PyTorch, TPU via PyTorch/XLA).

    Sessions that need dynamic or variant behaviour define their own subclasses
    or wrapper classes locally.
    """
    def __init__(
        self,
        d_model: int = D_MODEL,
        nhead: int = N_HEAD,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn  = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.attn(x, x, x)[0])
        x = self.norm2(x + self.ff(x))
        return x
