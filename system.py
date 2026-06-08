# system.py
# HW4 Part 2 – End-to-End Interactive AWGN Communication System
#
# Combines the TX encoder, AWGN channel, feedback channel, and RX decoder
# into a single differentiable module that can be trained end-to-end.
#
# Protocol for T rounds:
#   For t = 1, …, T:
#     1. TX encoder produces x^(t) from message m and history
#     2. x^(t) is transmitted through the AWGN forward channel → y^(t)
#     3. y^(t) is relayed back via the noiseless feedback channel → f^(t)
#   After T rounds:
#     4. RX decoder processes all y^(1), …, y^(T) → symbol logits → m̂
#
# Loss: sum of CrossEntropyLoss over all MSG_LEN symbol positions.
# Metric: Symbol Error Rate (SER) and Block Error Rate (BLER).

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from channel import AWGNChannel, FeedbackChannel
from encoder import TXEncoder
from decoder import RXDecoder

# Suppress norm_first nested tensor warning (cosmetic only)
warnings.filterwarnings("ignore", message="enable_nested_tensor")


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end system
# ─────────────────────────────────────────────────────────────────────────────

class CommunicationSystem(nn.Module):
    """
    End-to-end interactive AWGN communication system trained jointly.

    The TX encoder and RX decoder are trained simultaneously via
    backpropagation through the entire T-round protocol. Gradients flow
    through the channel (AWGN noise is differentiable w.r.t. x since
    y = x + ε and ε does not depend on x).

    Parameters
    ----------
    sigma2 : float  –  AWGN noise variance (default: config.SIGMA2 = 0.25)
    T      : int    –  number of communication rounds (default: config.T = 4)
    """

    def __init__(
        self,
        sigma2: float = config.SIGMA2,
        T:      int   = config.T,
    ):
        super().__init__()
        self.T       = T
        self.encoder = TXEncoder()
        self.decoder = RXDecoder()
        self.channel  = AWGNChannel(sigma2=sigma2)
        self.feedback = FeedbackChannel()

    def forward(self, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the full T-round interactive communication protocol.

        Parameters
        ----------
        m : (batch,)  –  integer message indices; each element is a flat index
                         in [0, ALPHABET^MSG_LEN) representing a full message.
                         Converted internally to per-position symbol indices.

        Returns
        -------
        logits : (batch, MSG_LEN, ALPHABET)  –  RX decoder output logits
        m_sym  : (batch, MSG_LEN)            –  per-position ground-truth labels
        """
        batch     = m.size(0)
        device    = m.device
        msg_len   = config.MSG_LEN
        alphabet  = config.ALPHABET

        # ── Convert flat message index → per-position symbol indices ──────────
        # m is a flat index in [0, 8^4); decode to 4 symbols in base 8.
        m_sym = torch.zeros(batch, msg_len, dtype=torch.long, device=device)
        m_tmp = m.clone()
        for pos in range(msg_len - 1, -1, -1):
            m_sym[:, pos] = m_tmp % alphabet
            m_tmp = m_tmp // alphabet

        # ── One-hot encode the message for the encoder ────────────────────────
        m_onehot = F.one_hot(m_sym, num_classes=alphabet).float()  # (batch, MSG_LEN, ALPHABET)

        # ── T-round interactive protocol ──────────────────────────────────────
        x_history: list[torch.Tensor] = []   # transmitted coded symbols
        f_history: list[torch.Tensor] = []   # received feedback signals
        y_list:    list[torch.Tensor] = []   # received signals at RX

        for t in range(self.T):
            # Step 1: TX encoder produces coded symbols for this round
            x_t = self.encoder(m_onehot, x_history, f_history)   # (batch, MSG_LEN)

            # Step 2: Forward AWGN channel
            y_t = self.channel(x_t)                               # (batch, MSG_LEN)

            # Step 3: Noiseless feedback to TX
            f_t = self.feedback(y_t)                              # (batch, MSG_LEN)

            # Accumulate history
            x_history.append(x_t)
            f_history.append(f_t)
            y_list.append(y_t)

        # ── RX decoder (executed once after all T rounds) ─────────────────────
        logits = self.decoder(y_list)   # (batch, MSG_LEN, ALPHABET)

        return logits, m_sym

    def compute_loss(
        self,
        logits: torch.Tensor,   # (batch, MSG_LEN, ALPHABET)
        m_sym:  torch.Tensor,   # (batch, MSG_LEN)
    ) -> torch.Tensor:
        """
        Compute the total CrossEntropyLoss summed over all MSG_LEN positions.

        Each symbol position is treated as an independent ALPHABET-class
        classification problem. The loss encourages the decoder to correctly
        identify each of the 4 transmitted symbols.

        Parameters
        ----------
        logits : (batch, MSG_LEN, ALPHABET)
        m_sym  : (batch, MSG_LEN)  –  ground-truth symbol indices

        Returns
        -------
        loss : scalar tensor
        """
        # CrossEntropyLoss expects (batch, classes, ...) or flatten
        # Reshape: (batch * MSG_LEN, ALPHABET) vs (batch * MSG_LEN,)
        batch, msg_len, alphabet = logits.shape
        loss = F.cross_entropy(
            logits.reshape(batch * msg_len, alphabet),
            m_sym.reshape(batch * msg_len),
        )
        return loss

    @torch.no_grad()
    def compute_metrics(
        self,
        logits: torch.Tensor,   # (batch, MSG_LEN, ALPHABET)
        m_sym:  torch.Tensor,   # (batch, MSG_LEN)
    ) -> dict[str, float]:
        """
        Compute Symbol Error Rate (SER) and Block Error Rate (BLER).

        SER  = fraction of individual symbols incorrectly decoded.
        BLER = fraction of complete messages where at least one symbol is wrong.

        Parameters
        ----------
        logits : (batch, MSG_LEN, ALPHABET)
        m_sym  : (batch, MSG_LEN)

        Returns
        -------
        dict with keys 'ser' and 'bler'
        """
        m_hat   = logits.argmax(dim=-1)              # (batch, MSG_LEN)
        correct = (m_hat == m_sym)                   # (batch, MSG_LEN) bool

        ser  = 1.0 - correct.float().mean().item()
        bler = 1.0 - correct.all(dim=-1).float().mean().item()

        return {"ser": ser, "bler": bler}


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    batch  = 32
    system = CommunicationSystem()

    enc_params = sum(p.numel() for p in system.encoder.parameters())
    dec_params = sum(p.numel() for p in system.decoder.parameters())
    total      = enc_params + dec_params
    print(f"TXEncoder params : {enc_params:,}")
    print(f"RXDecoder params : {dec_params:,}")
    print(f"Total    params  : {total:,}")

    # Random messages: flat indices in [0, 8^4)
    n_messages = config.ALPHABET ** config.MSG_LEN   # 4096
    m = torch.randint(0, n_messages, (batch,))

    # Forward pass
    logits, m_sym = system(m)
    print(f"\nlogits shape : {logits.shape}  (batch, MSG_LEN, ALPHABET)")
    print(f"m_sym  shape : {m_sym.shape}   (batch, MSG_LEN)")

    # Loss
    loss = system.compute_loss(logits, m_sym)
    print(f"\nCross-entropy loss : {loss.item():.4f}  "
          f"(random init → should be ≈ log({config.ALPHABET}) = {torch.log(torch.tensor(config.ALPHABET)).item():.4f})")

    # Metrics
    metrics = system.compute_metrics(logits, m_sym)
    print(f"SER  (random init) : {metrics['ser']:.4f}  (should be ≈ {(config.ALPHABET-1)/config.ALPHABET:.4f})")
    print(f"BLER (random init) : {metrics['bler']:.4f}  (should be ≈ 1.0)")

    # Backward pass check
    loss.backward()
    print(f"\n✅ system.py sanity check passed (gradients flow through T={config.T} rounds).")
