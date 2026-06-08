# train.py
# HW4 Part 2 – Training Loop for Interactive AWGN Communication System
#
# Trains the TX encoder and RX decoder end-to-end by minimising the
# CrossEntropyLoss over all MSG_LEN symbol positions simultaneously.
#
# Training details:
#   - Messages are sampled uniformly from {0, …, ALPHABET^MSG_LEN - 1}
#     at each mini-batch (online data generation, no fixed dataset).
#   - AdamW optimiser with ReduceLROnPlateau scheduler.
#   - Early stopping on validation SER.
#   - Best checkpoint saved to checkpoints/ (CHECKPOINT_DIR2 in config.py).
#
# Usage:
#   python train.py
#   python train.py --epochs 200 --batch_size 512

import os
import json
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from system import CommunicationSystem


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sample_messages(batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Sample a batch of random messages uniformly from the full message space
    {0, …, ALPHABET^MSG_LEN - 1}.

    Unlike Part 1 where data was loaded from a fixed dataset, here messages
    are generated on-the-fly at each training step. This ensures the model
    sees all possible messages over the course of training and prevents
    overfitting to a fixed training set.

    Parameters
    ----------
    batch_size : number of messages to sample
    device     : compute device

    Returns
    -------
    m : (batch_size,)  –  flat message indices
    """
    n_messages = config.ALPHABET ** config.MSG_LEN   # 8^4 = 4096
    return torch.randint(0, n_messages, (batch_size,), device=device)


# ─────────────────────────────────────────────────────────────────────────────
# One epoch
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    system:    CommunicationSystem,
    optimizer: torch.optim.Optimizer | None,
    device:    torch.device,
    n_batches: int,
    batch_size: int,
    training:  bool,
) -> dict[str, float]:
    """
    Run one epoch of training or evaluation.

    Each epoch consists of n_batches steps, each with a freshly sampled
    batch of random messages. Metrics are averaged over all steps.

    Parameters
    ----------
    system     : end-to-end CommunicationSystem
    optimizer  : AdamW (None during evaluation)
    device     : compute device
    n_batches  : number of mini-batches per epoch
    batch_size : samples per mini-batch
    training   : True for training, False for evaluation

    Returns
    -------
    dict with 'loss', 'ser', 'bler' averaged over the epoch
    """
    system.train() if training else system.eval()

    total_loss = 0.0
    total_ser  = 0.0
    total_bler = 0.0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for _ in range(n_batches):
            m = sample_messages(batch_size, device)

            logits, m_sym = system(m)
            loss = system.compute_loss(logits, m_sym)

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(system.parameters(), max_norm=1.0)
                optimizer.step()

            metrics = system.compute_metrics(logits, m_sym)
            total_loss += loss.item()
            total_ser  += metrics["ser"]
            total_bler += metrics["bler"]

    return {
        "loss": total_loss / n_batches,
        "ser" : total_ser  / n_batches,
        "bler": total_bler / n_batches,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(
    epochs:     int   = config.EPOCHS,
    batch_size: int   = config.BATCH_SIZE,
    lr:         float = config.LR,
    patience:   int   = config.PATIENCE,
    n_train:    int   = 200,
    n_val:      int   = 50,
    cfg_name:   str   = "upgraded",
) -> tuple[dict, str]:
    """
    Train the end-to-end communication system.

    Training procedure:
      1. Sample random messages uniformly at each step.
      2. Run the T-round interactive protocol (forward pass).
      3. Compute CrossEntropyLoss over all symbol positions.
      4. Backpropagate through the entire protocol (encoder + decoder jointly).
      5. Update with AdamW; step ReduceLROnPlateau on validation SER.
      6. Save best checkpoint (lowest val SER); stop early if no improvement.

    Parameters
    ----------
    epochs     : maximum training epochs
    batch_size : messages per mini-batch
    lr         : initial learning rate
    patience   : early-stopping patience
    n_train    : mini-batches per training epoch
    n_val      : mini-batches per validation epoch

    Returns
    -------
    history  : dict with 'train' and 'val' metric lists
    ckpt_path: path to the best checkpoint
    """
    os.makedirs(config.CHECKPOINT_DIR2, exist_ok=True)
    os.makedirs(config.RESULTS_DIR2,    exist_ok=True)

    device = get_device()
    print(f"[device] {device}")

    # Build system with selected config preset
    from encoder import TXEncoder
    from decoder import RXDecoder
    cfg = config.get_preset(cfg_name)
    system = CommunicationSystem().to(device)
    # Split cfg: encoder uses enc_hidden, decoder uses dec_hidden
    enc_cfg = {k: v for k, v in cfg.items() if k != "dec_hidden"}
    dec_cfg = {k: v for k, v in cfg.items() if k != "enc_hidden"}
    system.encoder = TXEncoder(**enc_cfg).to(device)
    system.decoder = RXDecoder(**dec_cfg).to(device)
    optimizer = AdamW(system.parameters(), lr=lr,
                      weight_decay=config.WEIGHT_DECAY)
    # Step on SER: lower is better
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    total_params = sum(p.numel() for p in system.parameters())
    print(f"\n{'═'*55}")
    print(f"  Interactive AWGN Communication  |  config: {cfg_name}  |  params: {total_params:,}")
    print(f"  T={config.T}  σ²={config.SIGMA2}  ALPHABET={config.ALPHABET}  MSG_LEN={config.MSG_LEN}")
    print(f"  {cfg}")
    print(f"{'═'*55}")

    ckpt_path = os.path.join(config.CHECKPOINT_DIR2, f"best_system_{cfg_name}.pt")
    best_ser  = float("inf")
    pat_count = 0
    history   = {"train": [], "val": []}

    for epoch in range(1, epochs + 1):
        # Training
        tr = run_epoch(system, optimizer, device,
                       n_batches=n_train, batch_size=batch_size, training=True)

        # Validation
        va = run_epoch(system, None, device,
                       n_batches=n_val, batch_size=batch_size, training=False)

        scheduler.step(va["ser"])

        history["train"].append(tr)
        history["val"].append(va)

        improved = va["ser"] < best_ser
        if improved:
            best_ser  = va["ser"]
            pat_count = 0
            torch.save({
                "epoch"     : epoch,
                "state_dict": system.state_dict(),
                "val_ser"   : best_ser,
                "val_bler"  : va["bler"],
                "cfg_name"  : cfg_name,
                "cfg"       : cfg,
            }, ckpt_path)
            tag = "✓"
        else:
            pat_count += 1
            tag = ""

        print(
            f"Epoch [{epoch:>3}/{epochs}]  "
            f"train loss: {tr['loss']:.4f}  ser: {tr['ser']:.4f}  bler: {tr['bler']:.4f}  |  "
            f"val loss: {va['loss']:.4f}  ser: {va['ser']:.4f}  bler: {va['bler']:.4f}  "
            f"lr: {optimizer.param_groups[0]['lr']:.2e}  {tag}"
        )

        if pat_count >= patience:
            print(f"\n[early stop] No improvement for {patience} epochs.")
            break

    # 1. Save history JSON
    import shutil
    hist_path = ckpt_path.replace(".pt", "_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f)

    # 2. Copy to default paths for test.py compatibility
    shutil.copy2(ckpt_path, os.path.join(config.CHECKPOINT_DIR2, "best_system.pt"))
    shutil.copy2(hist_path, os.path.join(config.CHECKPOINT_DIR2, "best_system_history.json"))

    print(f"\n[done] Best val SER: {best_ser:.4f}  →  {ckpt_path}")
    return history, ckpt_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="HW4 Part 2 Training – Interactive AWGN Communication"
    # Usage: python train.py
    )
    parser.add_argument("--epochs",      type=int,   default=config.EPOCHS)
    parser.add_argument("--batch_size",  type=int,   default=config.BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=config.LR)
    parser.add_argument("--patience",    type=int,   default=config.PATIENCE)
    parser.add_argument("--n_train",     type=int,   default=200,
                        help="Mini-batches per training epoch")
    parser.add_argument("--n_val",       type=int,   default=50,
                        help="Mini-batches per validation epoch")
    parser.add_argument("--config", type=str, default="upgraded",
                        choices=["baseline", "upgraded"],
                        help="Model config: baseline (d=64,L=2) or upgraded (d=128,L=4)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
        patience   = args.patience,
        n_train    = args.n_train,
        n_val      = args.n_val,
        cfg_name   = args.config,
    )