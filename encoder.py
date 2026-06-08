# encoder.py
# HW4 Part 2 – TX Encoder (Transformer-based)
#
# The transmitter (TX) encoder maps the original message m and the accumulated
# communication history (previously transmitted coded symbols and received
# feedback) to a new set of coded symbols x^(t) at each round t.
#
# Architecture (following HW Hint 3):
#   1. Pre-processing MLP  : maps raw input features → d_model
#   2. Transformer block(s): standard multi-head self-attention + FFN
#   3. Post-processing MLP : maps d_model → 1 (one coded scalar per symbol)
#   4. Power normalisation : enforces E[||x^(t)||²] ≤ 1
#
# Input at round t for each of the 4 symbol positions:
#   - One-hot encoded original symbol  (ALPHABET dims)
#   - All previously transmitted coded symbols  (t-1 values)
#   - All previously received feedback signals  (t-1 values)
#   Total raw feature dim per position = ALPHABET + 2*(t-1)  (varies by round)
#
# To keep the sequence length fixed at MSG_LEN = 4 across all rounds, the
# input is zero-padded to a fixed dimension before the pre-processing MLP.

import torch
import torch.nn as nn
import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from channel import power_normalise


# ─────────────────────────────────────────────────────────────────────────────
# Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding added to the input embeddings.
    Allows the Transformer to distinguish the four symbol positions.
    """

    def __init__(self, d_model: int, max_len: int = 16):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ─────────────────────────────────────────────────────────────────────────────
# TX Encoder
# ─────────────────────────────────────────────────────────────────────────────

class TXEncoder(nn.Module):
    """
    Transformer-based TX encoder for the interactive AWGN communication system.

    At each round t ∈ {1, …, T}, the encoder receives:
        - The original message m  (as one-hot vectors, shape (batch, MSG_LEN, ALPHABET))
        - Previously transmitted coded symbols x^(1), …, x^(t-1)
        - Previously received feedback signals  f^(1), …, f^(t-1)

    These are concatenated along the feature dimension for each symbol position,
    zero-padded to a fixed size (ALPHABET + 2*T), and projected to d_model via
    a pre-processing MLP. The Transformer then produces contextualised
    representations, which are mapped to one coded scalar per symbol position
    via a post-processing MLP. Power normalisation is applied before returning.

    Architecture summary:
        Input features per symbol  →  Pre-MLP  →  (batch, MSG_LEN, d_model)
        + Positional Encoding
        →  Transformer (N_LAYERS blocks)
        →  Post-MLP  →  (batch, MSG_LEN, 1)
        →  squeeze + power normalise  →  (batch, MSG_LEN)

    Parameters
    ----------
    alphabet    : number of possible symbols per position (default: 8)
    msg_len     : message length in symbols (default: 4)
    T           : total number of communication rounds (default: 4)
    d_model     : transformer embedding dimension
    n_heads     : number of attention heads
    n_layers    : number of transformer encoder blocks
    dim_ff      : feed-forward hidden dimension
    dropout     : dropout probability
    enc_hidden  : pre/post MLP hidden dimension
    """

    def __init__(
        self,
        alphabet   : int   = config.ALPHABET,
        msg_len    : int   = config.MSG_LEN,
        T          : int   = config.T,
        d_model    : int   = config.D_MODEL,
        n_heads    : int   = config.N_HEADS,
        n_layers   : int   = config.N_LAYERS,
        dim_ff     : int   = config.DIM_FF,
        dropout    : float = config.DROPOUT,
        enc_hidden : int   = config.ENC_HIDDEN,
    ):
        super().__init__()
        self.msg_len   = msg_len
        self.T         = T
        self.alphabet  = alphabet

        # Fixed input feature size per symbol position:
        # one-hot(ALPHABET) + T coded scalars (padded) + T feedback scalars (padded)
        self.input_dim = alphabet + 2 * T

        # ── Pre-processing MLP (maps raw features → d_model) ──────────────────
        # As specified in HW Hint 3: one MLP before the Transformer.
        self.pre_mlp = nn.Sequential(
            nn.Linear(self.input_dim, enc_hidden),
            nn.ReLU(),
            nn.Linear(enc_hidden, d_model),
        )

        # ── Positional encoding ───────────────────────────────────────────────
        self.pos_enc = PositionalEncoding(d_model, max_len=msg_len)

        # ── Transformer encoder ───────────────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model              = d_model,
            nhead                = n_heads,
            dim_feedforward      = dim_ff,
            dropout              = dropout,
            batch_first          = True,   # (batch, seq, d_model)
            norm_first           = True,   # Pre-LN: more stable training
        )
        # Disable nested tensor optimisation: incompatible with norm_first=True
        # in some PyTorch versions, causing a harmless but noisy UserWarning.
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers,
                                                    enable_nested_tensor=False)

        # ── Post-processing MLP (maps d_model → 1 coded scalar per position) ──
        # As specified in HW Hint 3: one MLP after the Transformer.
        self.post_mlp = nn.Sequential(
            nn.Linear(d_model, enc_hidden),
            nn.ReLU(),
            nn.Linear(enc_hidden, 1),
        )

    def forward(
        self,
        m_onehot  : torch.Tensor,          # (batch, MSG_LEN, ALPHABET)
        x_history : list[torch.Tensor],    # list of t-1 tensors, each (batch, MSG_LEN)
        f_history : list[torch.Tensor],    # list of t-1 tensors, each (batch, MSG_LEN)
    ) -> torch.Tensor:
        """
        Produce coded symbols for the current round.

        Parameters
        ----------
        m_onehot  : one-hot encoded original message, shape (batch, MSG_LEN, ALPHABET)
        x_history : list of previously transmitted coded symbol tensors
        f_history : list of previously received feedback tensors

        Returns
        -------
        x_t : (batch, MSG_LEN)  –  power-normalised coded symbols for this round
        """
        batch = m_onehot.size(0)

        # ── Build input feature tensor ────────────────────────────────────────
        # Start with one-hot message: (batch, MSG_LEN, ALPHABET)
        features = [m_onehot]

        # Append coded symbol history, zero-padded to T slots
        for t in range(self.T):
            if t < len(x_history):
                features.append(x_history[t].unsqueeze(-1))   # (batch, MSG_LEN, 1)
            else:
                features.append(torch.zeros(batch, self.msg_len, 1,
                                            device=m_onehot.device))

        # Append feedback history, zero-padded to T slots
        for t in range(self.T):
            if t < len(f_history):
                features.append(f_history[t].unsqueeze(-1))   # (batch, MSG_LEN, 1)
            else:
                features.append(torch.zeros(batch, self.msg_len, 1,
                                            device=m_onehot.device))

        # Concatenate along feature dimension: (batch, MSG_LEN, ALPHABET + 2*T)
        z = torch.cat(features, dim=-1)

        # ── Pre-processing MLP ─────────────────────────────────────────────────
        z = self.pre_mlp(z)               # (batch, MSG_LEN, d_model)

        # ── Positional encoding ────────────────────────────────────────────────
        z = self.pos_enc(z)               # (batch, MSG_LEN, d_model)

        # ── Transformer ────────────────────────────────────────────────────────
        z = self.transformer(z)           # (batch, MSG_LEN, d_model)

        # ── Post-processing MLP ────────────────────────────────────────────────
        x_t = self.post_mlp(z).squeeze(-1)   # (batch, MSG_LEN)

        # ── Power normalisation ────────────────────────────────────────────────
        x_t = power_normalise(x_t)        # (batch, MSG_LEN)

        return x_t


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import torch.nn.functional as F

    batch   = 8
    encoder = TXEncoder()
    total   = sum(p.numel() for p in encoder.parameters())
    print(f"TXEncoder  params: {total:,}")
    print(f"Input dim per symbol: {encoder.input_dim}  "
          f"(ALPHABET={config.ALPHABET} + 2×T={2*config.T})")

    # Simulate a random message
    m = torch.randint(0, config.ALPHABET, (batch, config.MSG_LEN))
    m_oh = F.one_hot(m, num_classes=config.ALPHABET).float()

    print(f"\nRound 1 (no history):")
    x1 = encoder(m_oh, x_history=[], f_history=[])
    print(f"  x1 shape        : {x1.shape}  (batch, MSG_LEN)")
    print(f"  x1 vector power : {x1.pow(2).sum(dim=-1).mean():.4f}  (should be ≈ 1.0)")

    print(f"\nRound 2 (with x1 history and f1 feedback):")
    f1 = x1 + torch.randn_like(x1) * config.SIGMA
    x2 = encoder(m_oh, x_history=[x1], f_history=[f1])
    print(f"  x2 shape        : {x2.shape}")
    print(f"  x2 vector power : {x2.pow(2).sum(dim=-1).mean():.4f}  (should be ≈ 1.0)")

    print(f"\nRound 4 (full history):")
    f2 = x2 + torch.randn_like(x2) * config.SIGMA
    x3 = encoder(m_oh, x_history=[x1, x2], f_history=[f1, f2])
    f3 = x3 + torch.randn_like(x3) * config.SIGMA
    x4 = encoder(m_oh, x_history=[x1, x2, x3], f_history=[f1, f2, f3])
    print(f"  x4 shape        : {x4.shape}")
    print(f"  x4 vector power : {x4.pow(2).sum(dim=-1).mean():.4f}  (should be ≈ 1.0)")

    print(f"\n✅ encoder.py sanity check passed.")