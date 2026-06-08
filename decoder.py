# decoder.py
# HW4 Part 2 – RX Decoder (Transformer-based)
#
# The receiver (RX) decoder collects all T noisy received signals y^(1), …, y^(T)
# and produces a symbol estimate m̂ ∈ {0, …, ALPHABET-1}^MSG_LEN.
#
# As specified in HW Hint 2, the decoder is executed only once at the end of
# all T communication rounds, after all coded symbols have been received.
#
# Architecture (following HW Hint 3):
#   1. Pre-processing MLP  : maps concatenated received signals → d_model
#   2. Transformer block(s): standard multi-head self-attention + FFN
#   3. Post-processing MLP : maps d_model → ALPHABET (logits per symbol position)
#
# Input: all T received signals y^(1), …, y^(T), stacked along the feature
# dimension for each of the MSG_LEN symbol positions.
# Input dim per position = T  (one received scalar per round per position)
#
# Output: logits of shape (batch, MSG_LEN, ALPHABET) — one distribution over
# the 8-symbol alphabet per position. CrossEntropyLoss is applied per position.

import torch
import torch.nn as nn
import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from encoder import PositionalEncoding


# ─────────────────────────────────────────────────────────────────────────────
# RX Decoder
# ─────────────────────────────────────────────────────────────────────────────

class RXDecoder(nn.Module):
    """
    Transformer-based RX decoder for the interactive AWGN communication system.

    After all T communication rounds, the decoder receives the complete set of
    noisy signals Y = [y^(1), …, y^(T)] ∈ R^(batch, MSG_LEN, T) and produces
    symbol estimates for each of the MSG_LEN positions independently.

    Architecture summary:
        Stacked received signals  →  Pre-MLP  →  (batch, MSG_LEN, d_model)
        + Positional Encoding
        →  Transformer (N_LAYERS blocks)
        →  Post-MLP  →  (batch, MSG_LEN, ALPHABET)   [logits]

    The logits are passed to nn.CrossEntropyLoss during training and to
    argmax at inference time to obtain symbol estimates m̂.

    Parameters
    ----------
    alphabet    : number of possible symbols per position (default: 8)
    msg_len     : message length in symbols (default: 4)
    T           : total number of communication rounds (default: 4)
    d_model     : transformer embedding dimension
    n_heads     : number of attention heads
    n_layers    : number of transformer decoder blocks
    dim_ff      : feed-forward hidden dimension
    dropout     : dropout probability
    dec_hidden  : pre/post MLP hidden dimension
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
        dec_hidden : int   = config.DEC_HIDDEN,
    ):
        super().__init__()
        self.msg_len  = msg_len
        self.T        = T
        self.alphabet = alphabet

        # Input feature dim per symbol position = T received scalars (one per round)
        self.input_dim = T

        # ── Pre-processing MLP (maps T received scalars → d_model) ────────────
        # As specified in HW Hint 3: one MLP before the Transformer.
        self.pre_mlp = nn.Sequential(
            nn.Linear(self.input_dim, dec_hidden),
            nn.ReLU(),
            nn.Linear(dec_hidden, d_model),
        )

        # ── Positional encoding ───────────────────────────────────────────────
        self.pos_enc = PositionalEncoding(d_model, max_len=msg_len)

        # ── Transformer encoder ───────────────────────────────────────────────
        # Note: we use a Transformer *encoder* (not decoder) here because the
        # RX has full access to all T received signals simultaneously (no
        # causal masking needed). The name "decoder" refers to its role in the
        # communication system, not to the Transformer variant used.
        enc_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = dim_ff,
            dropout         = dropout,
            batch_first     = True,   # (batch, seq, d_model)
            norm_first      = True,   # Pre-LN for stable training
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers,
                                                    enable_nested_tensor=False)

        # ── Post-processing MLP (maps d_model → ALPHABET logits per position) ─
        # As specified in HW Hint 3: one MLP after the Transformer.
        self.post_mlp = nn.Sequential(
            nn.Linear(d_model, dec_hidden),
            nn.ReLU(),
            nn.Linear(dec_hidden, alphabet),
        )

    def forward(self, y_list: list[torch.Tensor]) -> torch.Tensor:
        """
        Decode the received signals from all T rounds into symbol logits.

        Parameters
        ----------
        y_list : list of T tensors, each of shape (batch, MSG_LEN)
                 y_list[t] = y^(t+1) = received signal at round t+1

        Returns
        -------
        logits : (batch, MSG_LEN, ALPHABET)
                 Raw (un-normalised) log-probabilities for each symbol position.
                 Apply CrossEntropyLoss during training, argmax at inference.
        """
        assert len(y_list) == self.T, (
            f"Expected {self.T} received signals, got {len(y_list)}"
        )

        # Stack received signals along feature dim:
        # each y^(t): (batch, MSG_LEN) → stack → (batch, MSG_LEN, T)
        Y = torch.stack(y_list, dim=-1)    # (batch, MSG_LEN, T)

        # ── Pre-processing MLP ─────────────────────────────────────────────────
        z = self.pre_mlp(Y)               # (batch, MSG_LEN, d_model)

        # ── Positional encoding ────────────────────────────────────────────────
        z = self.pos_enc(z)               # (batch, MSG_LEN, d_model)

        # ── Transformer ────────────────────────────────────────────────────────
        z = self.transformer(z)           # (batch, MSG_LEN, d_model)

        # ── Post-processing MLP ────────────────────────────────────────────────
        logits = self.post_mlp(z)         # (batch, MSG_LEN, ALPHABET)

        return logits

    def decode(self, y_list: list[torch.Tensor]) -> torch.Tensor:
        """
        Convenience method: returns predicted symbol indices (argmax of logits).

        Parameters
        ----------
        y_list : list of T tensors, each (batch, MSG_LEN)

        Returns
        -------
        m_hat : (batch, MSG_LEN)  –  predicted symbol indices ∈ {0, …, ALPHABET-1}
        """
        logits = self.forward(y_list)          # (batch, MSG_LEN, ALPHABET)
        return logits.argmax(dim=-1)           # (batch, MSG_LEN)


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    batch   = 8
    decoder = RXDecoder()
    total   = sum(p.numel() for p in decoder.parameters())
    print(f"RXDecoder  params: {total:,}")
    print(f"Input dim per symbol: {decoder.input_dim}  (T={config.T} rounds)")
    print(f"Output dim per symbol: {decoder.alphabet}  (ALPHABET={config.ALPHABET})")

    # Simulate T received signals
    y_list = [
        torch.randn(batch, config.MSG_LEN)
        for _ in range(config.T)
    ]

    # Forward pass
    logits = decoder(y_list)
    print(f"\nlogits shape : {logits.shape}  "
          f"(batch={batch}, MSG_LEN={config.MSG_LEN}, ALPHABET={config.ALPHABET})")

    # Decode
    m_hat = decoder.decode(y_list)
    print(f"m_hat  shape : {m_hat.shape}  (batch, MSG_LEN)")
    print(f"m_hat sample : {m_hat[0].tolist()}  "
          f"(values should be in [0, {config.ALPHABET-1}])")

    # Check logits are valid (finite, correct shape)
    assert logits.shape == (batch, config.MSG_LEN, config.ALPHABET)
    assert m_hat.shape  == (batch, config.MSG_LEN)
    assert m_hat.min() >= 0 and m_hat.max() < config.ALPHABET

    print(f"\n✅ decoder.py sanity check passed.")
