# Interactive AWGN Communication with Transformer

**CS515 Deep Learning | Homework 4, Part 2 (Bonus)**
Sabancı University

A Transformer-based end-to-end learned communication system over an Additive White Gaussian Noise (AWGN) channel with noiseless feedback. The transmitter (TX) and receiver (RX) are jointly trained to learn an effective interactive communication protocol from scratch.

---

## Problem Description

A two-node interactive communication system where:

- **TX** holds a message $m \in \{1, \ldots, 8\}^4$ — a sequence of 4 symbols drawn from an 8-symbol alphabet (4096 possible messages)

> **Implementation note:** The assignment symbols `{1,...,8}` are internally represented as class indices `{0,...,7}` for compatibility with PyTorch `CrossEntropyLoss`. This is a standard zero-indexing convention and does not affect the communication protocol.
- **Forward channel:** AWGN — $y^{(t)} = x^{(t)} + \varepsilon^{(t)}$, $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$
- **Feedback channel:** Noiseless relay — $f^{(t)} = y^{(t)}$
- **Goal:** RX reconstructs $\hat{m}$ correctly after $T = 4$ communication rounds

Both TX encoder and RX decoder are Transformer-based neural networks trained end-to-end via backpropagation through the entire $T$-round protocol.

---

## Results

### Upgraded Configuration (d=128, L=4, H=8)

| Metric | Value |
|---|---|
| Symbol Error Rate (SER) | **29.16%** |
| Block Error Rate (BLER) | **69.70%** |
| Training SNR | 0 dB (vector-level: E[||x||²]=1, E[||ε||²]=MSG_LEN×σ²=1) |
| Training epochs | 65 (early stop) |

### Baseline vs Upgraded Comparison

| Config | Params | Val SER | Val BLER | Test / 0 dB SER | Test / 0 dB BLER |
|---|---:|---:|---:|---:|---:|
| Baseline (d=64, L=2, H=4) | 152,521 | 28.97% | 69.77% | 29.16%* | 70.08%* |
| Upgraded (d=128, L=4, H=8) | 1,129,865 | **28.96%** | **69.03%** | **29.16%** | **69.70%** |

\* Baseline test SER/BLER are taken from the 0 dB point of the SNR comparison sweep in the latest run.

The validation performance gap is marginal, and the test performance of the two configurations remains very close. In the latest run, the upgraded model achieves 29.16% SER and 69.70% BLER, while the baseline remains in the same approximately 29% SER range. At the vector-level training SNR of 0 dB, increasing model capacity does not provide a meaningful performance gain, suggesting that the noisy channel and limited communication budget (T=4 rounds, code rate R=1) are stronger bottlenecks than model capacity.

### Per-position SER (Upgraded)

| Position | SER |
|---|---:|
| Position 1 | 29.85% |
| Position 2 | 29.19% |
| Position 3 | 28.61% |
| Position 4 | 29.01% |

The per-position errors are close to the overall SER, indicating that the model does not strongly favor or neglect any specific symbol position.

### SER vs SNR

SNR values use the vector-level convention: SNR = E[||x||²] / E[||ε||²] = 1 / (MSG_LEN × σ²).
The σ² column shows the per-dimension noise variance used in the AWGN channel.
The SNR sweep is evaluated with randomly sampled test messages at each SNR point, so the 0 dB value may differ slightly from the main test-set estimate above.

| SNR (dB) | σ² | Upgraded SER | Upgraded BLER | Baseline SER | Baseline BLER |
|---|---:|---:|---:|---:|---:|
| −5 | 0.7906 | 60.84% | 96.42% | 60.44% | 96.24% |
| −3 | 0.4988 | 50.28% | 91.20% | 50.06% | 90.92% |
| −1 | 0.3147 | 36.68% | 79.94% | 36.48% | 79.14% |
| **0** | **0.2500** | **29.36%** | **70.36%** | **29.16%** | **70.08%** |
| 1 | 0.1986 | 22.19% | 58.84% | 22.04% | 58.30% |
| 2 | 0.1577 | 15.17% | 43.86% | 15.72% | 46.56% |
| 3 | 0.1253 | 9.31% | 30.70% | 10.16% | 32.48% |
| 5 | 0.0791 | 2.80% | 10.16% | 3.21% | 11.90% |
| 7 | 0.0499 | 0.47% | 1.78% | 0.74% | 2.88% |
| 10 | 0.0250 | 0.05% | 0.20% | 0.13% | 0.54% |

---

## Project Structure

```
interactive-awgn-communication/
│
├── config.py           # All hyperparameters (T, σ², alphabet, d_model, presets)
├── channel.py          # Power normalisation, AWGN forward channel, noiseless feedback
├── encoder.py          # TX Transformer encoder (pre-MLP → Transformer → post-MLP)
├── decoder.py          # RX Transformer decoder (executed once after all T rounds)
├── system.py           # End-to-end system: T-round protocol, loss, SER/BLER metrics
├── train.py            # Training loop (AdamW + early stopping + online message sampling)
├── test.py             # Evaluation: SER/BLER, per-position SER, SER vs SNR sweep,
│                       #             confusion matrix, constellation plots,
│                       #             baseline vs upgraded comparison
├── requirements.txt
├── README.md
│
├── checkpoints/        # .pt checkpoint files are generated automatically during training.
│   │                   # Pretrained model weights are not included in the submission.
│   │                   # Run training first before evaluation.
│   ├── best_system_history.json
│   ├── best_system_baseline_history.json
│   └── best_system_upgraded_history.json
│
└── results/            # Generated figures 
    ├── p2_training_curves.png
    ├── p2_per_position_ser.png
    ├── p2_ser_vs_snr.png
    ├── p2_confusion_matrix_upgraded.png
    ├── p2_constellation_r1_upgraded.png
    ├── p2_constellation_r2_upgraded.png
    ├── p2_constellation_r3_upgraded.png
    ├── p2_constellation_r4_upgraded.png
    ├── p2_baseline_vs_upgraded.png
    └── p2_baseline_vs_upgraded_snr.png
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

### Train

Note: Pretrained `.pt` checkpoint files are not included in this repository.
Before running `test.py`, train the model first:

```bash
# Train baseline config
python train.py --config baseline --epochs 100

# Train upgraded config (default)
python train.py --config upgraded --epochs 100

# Quick test (3 epochs)
python train.py --config upgraded --epochs 3 --n_train 20 --n_val 10
```

### Evaluate

```bash
# Full evaluation with all figures
python test.py --n_test 10000

# With baseline vs upgraded comparison
python test.py --n_test 10000 --compare

# Skip SNR sweep (faster)
python test.py --n_test 10000 --skip_snr
```

---

## Architecture

### TX Encoder (runs at each round t = 1, …, T)

```
Input: original message (one-hot) + coded symbol history + feedback history
         ↓
Pre-processing MLP  →  (batch, MSG_LEN, d_model)
         ↓
Positional Encoding
         ↓
Transformer Encoder (N_LAYERS blocks, N_HEADS attention heads)
         ↓
Post-processing MLP  →  (batch, MSG_LEN, 1)
         ↓
Power Normalisation  →  x^(t) ∈ R^(batch, MSG_LEN),  E[||x||²] ≤ 1
```

### RX Decoder (runs once after all T rounds)

```
Input: stacked received signals [y^(1), …, y^(T)] ∈ R^(batch, MSG_LEN, T)
         ↓
Pre-processing MLP  →  (batch, MSG_LEN, d_model)
         ↓
Positional Encoding
         ↓
Transformer Encoder (N_LAYERS blocks)
         ↓
Post-processing MLP  →  (batch, MSG_LEN, ALPHABET)  [logits]
         ↓
argmax  →  m̂ ∈ {0, …, ALPHABET-1}^MSG_LEN
```

---

## Hyperparameters

| Parameter | Baseline | Upgraded |
|---|---|---|
| d_model | 64 | **128** |
| N_HEADS | 4 | **8** |
| N_LAYERS | 2 | **4** |
| DIM_FF | 128 | **256** |
| T (rounds) | 4 | 4 |
| σ² (noise) | 0.25 | 0.25 |
| ALPHABET | 8 | 8 |
| MSG_LEN | 4 | 4 |
| Batch size | 256 | 256 |
| Optimizer | AdamW | AdamW |

---

## Key Design Choices

- **Online message sampling** — messages are sampled uniformly from all 4096 possibilities at each training step, ensuring full coverage without a fixed dataset
- **Power constraint** — each transmitted 4-dimensional coded vector is normalised to approximately unit squared norm, satisfying E[||x^(t)||²] ≤ 1 (vector-level SNR = 0 dB since E[||ε||²] = MSG_LEN × σ² = 4 × 0.25 = 1)
- **Noiseless feedback** — received signal $y^{(t)}$ is relayed back to TX at zero cost (Hint 1)
- **Decoder executed once** — all $T$ received signals are collected before decoding (Hint 2)
- **Pre-LN Transformer** — `norm_first=True` for more stable training
- **Gradient clipping** — `max_norm=1.0` applied at each step
- **Early stopping** — on validation SER with patience=15
