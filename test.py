# test.py
# HW4 Part 2 – Evaluation of the Interactive AWGN Communication System
#
# Loads the best trained checkpoint and evaluates on a large test set of
# randomly sampled messages. Reports:
#   - Symbol Error Rate (SER)
#   - Block Error Rate (BLER)
#   - Per-position SER  (does the model struggle more on certain positions?)
#   - SER vs SNR curve  (evaluates robustness across different noise levels)
#
# All figures are saved to results/.
#
# Usage:
#   python test.py
#   python test.py --n_test 10000

import os
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from channel import AWGNChannel
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


def load_system(device: torch.device,
                ckpt_path: str = None) -> CommunicationSystem:
    """
    Load a checkpoint saved by train.py.

    If the checkpoint contains a 'cfg' key, the model is rebuilt from those
    hyperparameters so the architecture always matches the saved weights.
    """
    from encoder import TXEncoder
    from decoder import RXDecoder

    if ckpt_path is None:
        ckpt_path = os.path.join(config.CHECKPOINT_DIR2, "best_system.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Run: python train.py"
        )
    ckpt = torch.load(ckpt_path, map_location=device)

    # Rebuild model from saved cfg if available, else fall back to upgraded preset
    cfg_name = ckpt.get("cfg_name", "upgraded")
    cfg      = ckpt.get("cfg", config.get_preset(cfg_name))
    system   = CommunicationSystem().to(device)
    enc_cfg  = {k: v for k, v in cfg.items() if k != "dec_hidden"}
    dec_cfg  = {k: v for k, v in cfg.items() if k != "enc_hidden"}
    system.encoder = TXEncoder(**enc_cfg).to(device)
    system.decoder = RXDecoder(**dec_cfg).to(device)

    system.load_state_dict(ckpt["state_dict"])
    system.eval()
    cfg_label = ckpt.get("cfg_name", "unknown")
    print(f"[loaded] {ckpt_path}  "
          f"(val SER: {ckpt['val_ser']:.4f}  "
          f"val BLER: {ckpt['val_bler']:.4f}  "
          f"@ epoch {ckpt['epoch']}  cfg: {cfg_label})")
    return system


def save_fig(fig: plt.Figure, filename: str) -> None:
    os.makedirs(config.RESULTS_DIR2, exist_ok=True)
    path = os.path.join(config.RESULTS_DIR2, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved]  {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    system:     CommunicationSystem,
    device:     torch.device,
    n_test:     int = 10_000,
    batch_size: int = 512,
) -> dict:
    """
    Evaluate the system on n_test randomly sampled messages.

    Computes:
    - Overall SER and BLER
    - Per-position SER (MSG_LEN values)

    Parameters
    ----------
    system     : trained CommunicationSystem in eval() mode
    device     : compute device
    n_test     : total number of test messages
    batch_size : evaluation batch size

    Returns
    -------
    dict with 'ser', 'bler', 'per_position_ser'
    """
    n_messages  = config.ALPHABET ** config.MSG_LEN
    all_correct = []   # (n_test, MSG_LEN) bool

    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        m   = torch.randint(0, n_messages, (end - start,), device=device)

        logits, m_sym = system(m)
        m_hat  = logits.argmax(dim=-1)          # (batch, MSG_LEN)
        correct = (m_hat == m_sym).cpu()        # (batch, MSG_LEN)
        all_correct.append(correct)

    all_correct = torch.cat(all_correct, dim=0).float()   # (n_test, MSG_LEN)

    ser             = 1.0 - all_correct.mean().item()
    bler            = 1.0 - all_correct.all(dim=-1).float().mean().item()
    per_pos_ser     = (1.0 - all_correct.mean(dim=0)).tolist()   # MSG_LEN values

    return {
        "ser"            : ser,
        "bler"           : bler,
        "per_position_ser": per_pos_ser,
        "n_test"         : n_test,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SER vs SNR sweep
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def ser_vs_snr(
    system:     CommunicationSystem,
    device:     torch.device,
    snr_db_range: list = [-5, -3, -1, 0, 1, 2, 3, 5, 7, 10],
    n_test:     int = 5_000,
    batch_size: int = 512,
) -> dict:
    """
    Evaluate SER and BLER across a range of SNR values.

    SNR (dB) = 10 * log10(signal power / noise power)
             = 10 * log10(1 / (MSG_LEN * σ²))
    under the vector-level power convention, since ||x||² is normalised
    to 1 and the total AWGN noise power is MSG_LEN * σ². For each SNR,
    a new CommunicationSystem is instantiated with the corresponding σ², the
    trained weights are loaded, and the system is evaluated.

    This shows how robust the learned communication protocol is to
    varying channel conditions beyond the training noise level.

    Parameters
    ----------
    system       : trained system (weights are reused at each SNR)
    device       : compute device
    snr_db_range : list of SNR values in dB to evaluate
    n_test       : messages per SNR point
    batch_size   : evaluation batch size

    Returns
    -------
    dict with 'snr_db', 'ser', 'bler' lists
    """
    # Training SNR for reference
    train_snr_db = 10 * np.log10(1.0 / (config.MSG_LEN * config.SIGMA2))

    ser_list  = []
    bler_list = []

    for snr_db in snr_db_range:
        # Vector-level SNR convention: SNR = 1 / (MSG_LEN × σ²)
        # → σ² = 1 / (MSG_LEN × 10^(SNR_dB/10))
        sigma2 = 1.0 / (config.MSG_LEN * (10 ** (snr_db / 10.0)))

        # Build a new system with this σ², copying architecture from system.
        # Use deepcopy to preserve the original system's architecture
        # (handles both baseline d=64 and upgraded d=128 transparently).
        import copy
        sys_test = copy.deepcopy(system)
        sys_test.channel = AWGNChannel(sigma2=sigma2)
        sys_test.to(device)
        sys_test.eval()

        n_messages = config.ALPHABET ** config.MSG_LEN
        correct_all = []

        for start in range(0, n_test, batch_size):
            end = min(start + batch_size, n_test)
            m   = torch.randint(0, n_messages, (end - start,), device=device)
            logits, m_sym = sys_test(m)
            m_hat   = logits.argmax(dim=-1)
            correct = (m_hat == m_sym).cpu()
            correct_all.append(correct)

        correct_all = torch.cat(correct_all, dim=0).float()
        ser_list.append(1.0 - correct_all.mean().item())
        bler_list.append(1.0 - correct_all.all(dim=-1).float().mean().item())

        print(f"  SNR = {snr_db:>5.1f} dB  (σ²={sigma2:.4f})  "
              f"SER={ser_list[-1]:.4f}  BLER={bler_list[-1]:.4f}"
              + ("  ← training SNR" if abs(snr_db - train_snr_db) < 0.5 else ""))

    return {
        "snr_db"       : snr_db_range,
        "ser"          : ser_list,
        "bler"         : bler_list,
        "train_snr_db" : train_snr_db,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves() -> None:
    """Plot train/val loss, SER, and BLER curves from saved history."""
    hist_path = os.path.join(config.CHECKPOINT_DIR2, "best_system_history.json")
    if not os.path.exists(hist_path):
        print("[skip] No history file found.")
        return

    with open(hist_path) as f:
        history = json.load(f)

    epochs = range(1, len(history["train"]) + 1)
    metrics = ["loss", "ser", "bler"]
    labels  = ["Cross-Entropy Loss", "Symbol Error Rate (SER)",
                "Block Error Rate (BLER)"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Part 2 – Training Curves: Interactive AWGN Communication",
                 fontsize=13)

    for ax, metric, label in zip(axes, metrics, labels):
        tr_vals = [ep[metric] for ep in history["train"]]
        va_vals = [ep[metric] for ep in history["val"]]
        ax.plot(epochs, tr_vals, label="Train", linewidth=2, color="#2563EB")
        ax.plot(epochs, va_vals, label="Val",   linewidth=2, color="#DC2626",
                linestyle="--")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    save_fig(fig, "p2_training_curves.png")


def plot_per_position_ser(metrics: dict) -> None:
    """Bar chart of SER per symbol position."""
    per_pos = metrics["per_position_ser"]
    x = range(1, len(per_pos) + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(x, per_pos, color="#2563EB", alpha=0.85)
    ax.bar_label(bars, fmt="%.4f", fontsize=9, padding=3)
    ax.axhline(metrics["ser"], color="red", linestyle="--", linewidth=1.5,
               label=f"Overall SER = {metrics['ser']:.4f}")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Position {i}" for i in x])
    ax.set_ylabel("Symbol Error Rate")
    ax.set_title("Part 2 – Per-Position SER on Test Set")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "p2_per_position_ser.png")


def plot_ser_vs_snr(snr_results: dict) -> None:
    """SER and BLER vs SNR curve."""
    snr_db = snr_results["snr_db"]
    train_snr = snr_results["train_snr_db"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Part 2 – SER & BLER vs SNR", fontsize=13)

    for ax, metric, label, color in zip(
        axes,
        ["ser", "bler"],
        ["Symbol Error Rate (SER)", "Block Error Rate (BLER)"],
        ["#2563EB", "#DC2626"],
    ):
        ax.semilogy(snr_db, snr_results[metric], "o-",
                    color=color, linewidth=2, markersize=6, label=label)
        ax.axvline(0.0, color="gray", linestyle="--", linewidth=1.5,
                   label="Training SNR = 0.0 dB  (σ²=0.25, ||x||²=1)")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    save_fig(fig, "p2_ser_vs_snr.png")



# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Baseline vs Upgraded comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_baseline_vs_upgraded(
    baseline_hist_path: str,
    upgraded_hist_path: str,
) -> None:
    """
    Overlay training/validation SER and BLER curves for baseline and upgraded
    configurations side by side on the same axes.

    Baseline  : d_model=64,  n_heads=4, n_layers=2
    Upgraded  : d_model=128, n_heads=8, n_layers=4
    """
    import json

    configs = {
        "Baseline (d=64, L=2)": baseline_hist_path,
        "Upgraded (d=128, L=4)": upgraded_hist_path,
    }
    colors = {"Baseline (d=64, L=2)": "#2563EB", "Upgraded (d=128, L=4)": "#DC2626"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Part 2 – Baseline vs Upgraded: Validation SER & BLER", fontsize=13)

    for label, hist_path in configs.items():
        if not os.path.exists(hist_path):
            print(f"[skip] {hist_path} not found.")
            continue
        with open(hist_path, "r", encoding="utf-8") as f:
            hist = json.load(f)
        epochs = range(1, len(hist["val"]) + 1)
        color  = colors[label]

        for ax, metric, title in zip(
            axes,
            ["ser", "bler"],
            ["Validation SER", "Validation BLER"],
        ):
            vals = [ep[metric] for ep in hist["val"]]
            ax.semilogy(epochs, vals, linewidth=2, color=color,
                        label=label)

    for ax, title in zip(axes, ["Validation SER", "Validation BLER"]):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    save_fig(fig, "p2_baseline_vs_upgraded.png")


def plot_baseline_vs_upgraded_snr(
    baseline_system: "CommunicationSystem",
    upgraded_system: "CommunicationSystem",
    device: torch.device,
    snr_db_range: list = [-5, -3, -1, 0, 1, 2, 3, 5, 7, 10],
    n_test: int = 5_000,
    batch_size: int = 512,
) -> None:
    """
    Plot SER vs SNR for baseline and upgraded systems on the same axes.
    Shows the performance gain from the architectural upgrade.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Part 2 – SER vs SNR: Baseline vs Upgraded", fontsize=13)

    configs = [
        ("Baseline (d=64, L=2)", baseline_system, "#2563EB"),
        ("Upgraded (d=128, L=4)", upgraded_system, "#DC2626"),
    ]

    for label, sys, color in configs:
        if sys is None:
            continue
        results = ser_vs_snr(sys, device, snr_db_range=snr_db_range,
                             n_test=n_test, batch_size=batch_size)
        ax.semilogy(snr_db_range, results["ser"], "o-",
                    color=color, linewidth=2, markersize=6, label=label)

    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1.5,
               label="Training SNR = 0.0 dB  (σ²=0.25, ||x||²=1)")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Symbol Error Rate (SER)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    save_fig(fig, "p2_baseline_vs_upgraded_snr.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: Confusion matrix (8×8)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def plot_confusion_matrix_p2(
    system:     "CommunicationSystem",
    device:     torch.device,
    n_test:     int = 10_000,
    batch_size: int = 512,
    label:      str = "Upgraded",
) -> None:
    """
    Plot the 8×8 confusion matrix for symbol classification.

    Each cell (i, j) shows how many times true symbol i was decoded as j.
    The diagonal should dominate for a well-trained system; off-diagonal
    entries reveal which symbols are most often confused.
    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    n_messages = config.ALPHABET ** config.MSG_LEN
    all_true, all_pred = [], []

    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        m   = torch.randint(0, n_messages, (end - start,), device=device)
        logits, m_sym = system(m)
        m_hat = logits.argmax(dim=-1)   # (batch, MSG_LEN)

        # Flatten all positions
        all_true.extend(m_sym.cpu().reshape(-1).tolist())
        all_pred.extend(m_hat.cpu().reshape(-1).tolist())

    cm   = confusion_matrix(all_true, all_pred,
                             labels=list(range(config.ALPHABET)))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[str(i) for i in range(config.ALPHABET)],
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format="d")
    ax.set_title(f"Part 2 – Symbol Confusion Matrix ({label})", fontsize=12)
    ax.set_xlabel("Predicted Symbol")
    ax.set_ylabel("True Symbol")
    fig.tight_layout()
    save_fig(fig, f"p2_confusion_matrix_{label.lower().replace(' ','_')}.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Coded symbol constellation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def plot_constellation(
    system:     "CommunicationSystem",
    device:     torch.device,
    n_samples:  int = 2_000,
    round_idx:  int = 0,
    label:      str = "Upgraded",
) -> None:
    """
    Scatter plot of the coded symbols x^(t) produced by the TX encoder
    at round `round_idx`, coloured by the first symbol of the message.

    This reveals whether the encoder has learned a symbol-dependent
    geometric structure in the signal space.

    Parameters
    ----------
    round_idx : which communication round to visualise (0-indexed)
    label     : config label for the filename
    """
    n_messages = config.ALPHABET ** config.MSG_LEN

    # Collect x^(t) at round_idx and the first symbol label
    all_x0, all_x1, all_sym = [], [], []

    batch_size = 256
    for _ in range(n_samples // batch_size + 1):
        m      = torch.randint(0, n_messages, (batch_size,), device=device)
        m_sym  = torch.zeros(batch_size, config.MSG_LEN, dtype=torch.long, device=device)
        m_tmp  = m.clone()
        for pos in range(config.MSG_LEN - 1, -1, -1):
            m_sym[:, pos] = m_tmp % config.ALPHABET
            m_tmp = m_tmp // config.ALPHABET

        import torch.nn.functional as F
        m_onehot = F.one_hot(m_sym, num_classes=config.ALPHABET).float()

        # Run encoder for round_idx rounds to get x at that round
        x_history, f_history = [], []
        for t in range(round_idx + 1):
            x_t = system.encoder(m_onehot, x_history, f_history)
            y_t = system.channel(x_t)
            f_t = system.feedback(y_t)
            x_history.append(x_t)
            f_history.append(f_t)

        x_t = x_history[round_idx].cpu()   # (batch, MSG_LEN)

        # Use position 0 and position 1 as 2D coordinates
        all_x0.extend(x_t[:, 0].tolist())
        all_x1.extend(x_t[:, 1].tolist())
        all_sym.extend(m_sym[:, 0].cpu().tolist())   # colour by first symbol

        if len(all_x0) >= n_samples:
            break

    all_x0  = np.array(all_x0[:n_samples])
    all_x1  = np.array(all_x1[:n_samples])
    all_sym = np.array(all_sym[:n_samples])

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")

    for sym in range(config.ALPHABET):
        mask = all_sym == sym
        ax.scatter(all_x0[mask], all_x1[mask],
                   s=15, alpha=0.5, color=cmap(sym),
                   label=f"Symbol {sym}")

    ax.set_title(
        f"Part 2 - Coded Symbol Constellation\n"
        f"Round t={round_idx+1}, Position (0,1)  ({label})",
        fontsize=11
    )
    ax.set_xlabel("x⁽ᵗ⁾ Position 0")
    ax.set_ylabel("x⁽ᵗ⁾ Position 1")
    ax.legend(fontsize=8, ncol=2, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(config.RESULTS_DIR2,
                        f"p2_constellation_r{round_idx+1}_{label.lower().replace(' ','_')}.png")
    os.makedirs(config.RESULTS_DIR2, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved]  {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="HW4 Part 2 Evaluation – Interactive AWGN Communication"
    # Usage: python test.py
    )
    parser.add_argument("--n_test",    type=int, default=10_000)
    parser.add_argument("--batch_size",type=int, default=512)
    parser.add_argument("--skip_snr",  action="store_true",
                        help="Skip the SER vs SNR sweep (faster evaluation)")
    parser.add_argument("--baseline_ckpt", type=str,
                        default="checkpoints/best_system_baseline.pt",
                        help="Path to baseline checkpoint for comparison plots")
    parser.add_argument("--compare", action="store_true",
                        help="Generate baseline vs upgraded comparison plots")
    return parser.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    device = get_device()
    system = load_system(device)  # loads best_system.pt with cfg from checkpoint

    # ── Main evaluation ───────────────────────────────────────────────────────
    print(f"\n[evaluating]  n_test={args.n_test:,}  ...")
    metrics = evaluate(system, device, n_test=args.n_test,
                       batch_size=args.batch_size)

    print(f"\n{'═'*50}")
    print(f"  Test Results  (n={metrics['n_test']:,})")
    print(f"{'═'*50}")
    print(f"  SER  : {metrics['ser']:.4f}  ({metrics['ser']*100:.2f}%)")
    print(f"  BLER : {metrics['bler']:.4f}  ({metrics['bler']*100:.2f}%)")
    print(f"\n  Per-position SER:")
    for i, s in enumerate(metrics["per_position_ser"], 1):
        print(f"    Position {i}: {s:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_training_curves()
    plot_per_position_ser(metrics)

    # ── SER vs SNR sweep ──────────────────────────────────────────────────────
    if not args.skip_snr:
        print(f"\n[SER vs SNR sweep]")
        snr_results = ser_vs_snr(system, device, n_test=5_000,
                                  batch_size=args.batch_size)
        plot_ser_vs_snr(snr_results)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    print(f"\n[confusion matrix]")
    plot_confusion_matrix_p2(system, device, n_test=args.n_test,
                             batch_size=args.batch_size, label="Upgraded")

    # ── Constellation ─────────────────────────────────────────────────────────
    print(f"\n[constellation plots]")
    for r in range(config.T):
        plot_constellation(system, device, n_samples=2_000,
                          round_idx=r, label="Upgraded")

    # ── Baseline vs Upgraded comparison ───────────────────────────────────────
    if args.compare:
        print(f"\n[baseline vs upgraded comparison]")
        baseline_hist = args.baseline_ckpt.replace(".pt", "_history.json")
        upgraded_hist = os.path.join(config.CHECKPOINT_DIR2,
                                     "best_system_upgraded_history.json")
        plot_baseline_vs_upgraded(baseline_hist, upgraded_hist)

        # Load baseline system for SNR comparison.
        # Baseline was trained with d_model=64, n_heads=4, n_layers=2 --
        # rebuild with those exact hyperparameters before loading weights.
        if os.path.exists(args.baseline_ckpt):
            # load_system reads cfg from checkpoint automatically
            baseline_system = load_system(device, ckpt_path=args.baseline_ckpt)
            print(f"\n[SNR comparison sweep]")
            plot_baseline_vs_upgraded_snr(
                baseline_system, system, device,
                n_test=5_000, batch_size=args.batch_size
            )
        else:
            print(f"[skip] Baseline checkpoint not found: {args.baseline_ckpt}")

    print(f"\n[done] All results saved to '{config.RESULTS_DIR2}/'")