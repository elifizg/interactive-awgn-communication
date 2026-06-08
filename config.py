# config.py
# HW4 Part 2 – Interactive AWGN Communication System
#
# All hyperparameters for the two-node communication protocol.
# Fixed by the assignment:
#   T = 4 communication rounds
#   σ² = 0.25 (noise variance)
#   Message alphabet: S = {1, …, 8}^4 (4 symbols from an 8-symbol alphabet)

# ── Channel ───────────────────────────────────────────────────────────────────
T           = 4          # number of communication rounds
SIGMA2      = 0.25       # AWGN noise variance σ²
SIGMA       = SIGMA2 ** 0.5   # noise std

# ── Message ───────────────────────────────────────────────────────────────────
ALPHABET    = 8          # |S| = 8 possible symbols per position
MSG_LEN     = 4          # message length (4 symbols)
# Total number of distinct messages = 8^4 = 4096

# ── Transformer architecture ──────────────────────────────────────────────────
# Upgraded from baseline (d_model=64, n_heads=4, n_layers=2):
#   d_model=128 doubles the embedding capacity.
#   n_heads=8   allows finer-grained multi-head attention (128/8=16 per head).
#   n_layers=4  adds depth for richer sequence representations.
#   dim_ff=256  scales the FFN proportionally with d_model.
D_MODEL     = 128        # embedding dimension (d_model)
N_HEADS     = 8          # number of attention heads  (d_model / N_HEADS = 16)
N_LAYERS    = 4          # number of transformer blocks
DIM_FF      = 256        # feed-forward hidden dimension (2 × d_model)
DROPOUT     = 0.1        # dropout in transformer

# ── MLP pre/post processing ───────────────────────────────────────────────────
# Encoder MLP maps raw input → d_model, then d_model → coded symbol (dim=1 per pos)
ENC_HIDDEN  = 128        # encoder MLP hidden size (scaled with d_model)
DEC_HIDDEN  = 128        # decoder MLP hidden size (scaled with d_model)

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE  = 256
EPOCHS      = 100
LR          = 1e-3
WEIGHT_DECAY= 1e-4
PATIENCE    = 15

# ── Output ────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR2 = "checkpoints"
RESULTS_DIR2    = "results"

# ── Architecture presets ──────────────────────────────────────────────────────

PRESETS = {
    "baseline": dict(
        d_model    = 64,
        n_heads    = 4,
        n_layers   = 2,
        dim_ff     = 128,
        enc_hidden = 64,
        dec_hidden = 64,
        description = "Baseline: d_model=64, n_heads=4, n_layers=2",
    ),
    "upgraded": dict(
        d_model    = 128,
        n_heads    = 8,
        n_layers   = 4,
        dim_ff     = 256,
        enc_hidden = 128,
        dec_hidden = 128,
        description = "Upgraded: d_model=128, n_heads=8, n_layers=4",
    ),
}


def get_preset(name: str) -> dict:
    """
    Return the hyperparameter dict for the given preset name.

    Parameters
    ----------
    name : 'baseline' or 'upgraded'

    Returns
    -------
    dict with keys: d_model, n_heads, n_layers, dim_ff, enc_hidden, dec_hidden
    """
    assert name in PRESETS, (
        f"Unknown preset '{name}'. Choose from: {list(PRESETS)}"
    )
    # Return a copy without the description key (not a model hyperparameter)
    cfg = {k: v for k, v in PRESETS[name].items() if k != "description"}
    return cfg


def print_presets() -> None:
    """Print a summary of all available presets."""
    print(f"\n{'─'*55}")
    print(f"  Available model presets:")
    print(f"{'─'*55}")
    for name, cfg in PRESETS.items():
        print(f"  [{name}]  {cfg['description']}")
        for k, v in cfg.items():
            if k != "description":
                print(f"    {k:<12} = {v}")
    print(f"{'─'*55}")


if __name__ == "__main__":
    print_presets()