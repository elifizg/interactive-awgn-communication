# part2/channel.py
# HW4 Part 2 – AWGN Channel and Power Constraint
#
# Implements:
#   1. Power normalisation: enforces E[||x||²] ≤ 1 per round.
#   2. AWGN forward channel: y = x + ε,  ε ~ N(0, σ²I).
#   3. Noiseless feedback channel: f = y  (simple relay, as per HW Hint 1).

import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def power_normalise(x: torch.Tensor) -> torch.Tensor:
    """
    Enforce the per-round average power constraint E[||x^(t)||²] ≤ 1.

    The coded symbols x ∈ R^(batch, MSG_LEN) are normalised so that the
    Euclidean norm of each transmitted vector equals exactly 1:

        x_norm = x / (||x|| + ε)

    This approximately enforces ||x_norm||² ≈ 1 for non-zero vectors,
    therefore satisfying the assignment constraint E[||x^(t)||²] <= 1.
    For zero vectors (edge case), the epsilon prevents division by zero
    and the output will be near-zero rather than exactly unit-norm. The small epsilon prevents division
    by zero. Normalisation is applied over the MSG_LEN=4 feature dimension
    (i.e. over the full 4-dimensional transmitted vector per sample).

    Parameters
    ----------
    x : (batch, MSG_LEN)  –  raw coded symbols from the encoder

    Returns
    -------
    (batch, MSG_LEN)  –  power-normalised coded symbols
    """
    # Compute per-sample power: ||x_i||² for each sample in the batch.
    # Use sum (not mean) so that the full vector norm is constrained to 1,
    # satisfying E[||x^(t)||²] <= 1 as required by the assignment.
    power = x.pow(2).sum(dim=-1, keepdim=True)   # (batch, 1)  = ||x||²
    # Approximately enforces ||x_norm||² ≈ 1 for non-zero vectors,
    # therefore satisfying E[||x^(t)||²] <= 1 as required by the assignment.
    return x / (power.sqrt() + 1e-8)


class AWGNChannel(nn.Module):
    """
    Additive White Gaussian Noise (AWGN) forward channel.

    Models the physical transmission medium between the TX encoder and the
    RX decoder. At each communication round t, the channel adds independent
    Gaussian noise to the transmitted signal:

        y^(t) = x^(t) + ε^(t),   ε^(t) ~ N(0, σ²I)

    where σ² is the noise variance specified in config.py. During evaluation
    (model.eval()), noise is still applied since the channel is always noisy.

    Parameters
    ----------
    sigma2 : float  –  noise variance σ² (default: config.SIGMA2 = 0.25)
    """

    def __init__(self, sigma2: float = config.SIGMA2):
        super().__init__()
        self.sigma = sigma2 ** 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transmit x through the AWGN channel.

        Parameters
        ----------
        x : (batch, MSG_LEN)  –  power-normalised coded symbols

        Returns
        -------
        y : (batch, MSG_LEN)  –  received signal after adding noise
        """
        noise = torch.randn_like(x) * self.sigma
        return x + noise


class FeedbackChannel(nn.Module):
    """
    Noiseless feedback channel (RX → TX).

    As suggested by HW Hint 1, the feedback signal is simply the noisy
    received signal y^(t) relayed back to the transmitter at zero cost.
    No additional noise is added on the feedback path.

    In the interactive communication protocol, this feedback allows the TX
    encoder to refine its coded symbols in subsequent rounds based on what
    the RX received in previous rounds.
    """

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Relay the received signal back to the transmitter.

        Parameters
        ----------
        y : (batch, MSG_LEN)  –  received signal from the forward channel

        Returns
        -------
        f : (batch, MSG_LEN)  –  feedback signal (identical to y)
        """
        return y   # noiseless relay: f^(t) = y^(t)


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    batch = 16
    msg_len = config.MSG_LEN

    # Simulate raw encoder output
    x_raw = torch.randn(batch, msg_len) * 3.0   # deliberately high power

    # Power normalise
    x_norm = power_normalise(x_raw)
    # E[||x||²]: expected squared norm over the batch (sum over MSG_LEN dim)
    power_before = x_raw.pow(2).sum(dim=-1).mean().item()
    power_after  = x_norm.pow(2).sum(dim=-1).mean().item()
    print(f"E[||x||²] before normalisation : {power_before:.4f}")
    print(f"E[||x||²] after  normalisation : {power_after:.4f}  (should be ≈ 1.0)")

    # Forward channel
    channel  = AWGNChannel()
    feedback = FeedbackChannel()

    y = channel(x_norm)
    f = feedback(y)

    import math
    # Vector-level SNR: signal power / noise power = ||x||² / (MSG_LEN × σ²) = 1 / (4 × 0.25) = 0 dB
    snr_db = 10 * math.log10(1.0 / (config.MSG_LEN * config.SIGMA2))
    print(f"\nAWGN channel  σ² = {config.SIGMA2}  vector-level SNR ≈ {snr_db:.2f} dB")
    print(f"  x shape : {x_norm.shape}")
    print(f"  y shape : {y.shape}")
    print(f"  f shape : {f.shape}")
    # E[||y-x||²] = E[||ε||²] = MSG_LEN × σ² (sum of 4 independent noise terms)
    print(f"  E[||y - x||²] : {(y - x_norm).pow(2).sum(dim=-1).mean().item():.4f}  "
          f"(should be ≈ MSG_LEN x sigma2 = {config.MSG_LEN * config.SIGMA2:.4f})")
    print(f"\n✅ channel.py sanity check passed.")