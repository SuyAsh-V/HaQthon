# IBM Qiskit Fall Fest 2026 — Hackathon Solutions

> **Team:** Hackathon Team &nbsp;|&nbsp; **Event:** IBM Qiskit Fall Fest 2026 &nbsp;|&nbsp; **Tracks:** 1 & 3

---

## Table of Contents

- [Track 1 — Error Detection via Spacetime Codes](#track-1--error-detection-via-spacetime-codes)
- [Track 3 — Tensor Network Surface Code Decoding](#track-3--tensor-network-surface-code-decoding)
- [Judging Criteria](#judging-criteria)
- [Full Explainer](#full-explainer)

---

## Track 1 — Error Detection via Spacetime Codes

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HzwvRaYCd4NOSVXM01e-Ec1RatQtl8IC?usp=sharing)

### Problem Statement

Standard Quantum Error Correction codes detect errors in space — taking a snapshot of all qubits at a single moment. **Spacetime codes** spread the detection process across both space and time using lightweight ancilla qubits (Controlled-Pauli checks), which is critical for near-term quantum devices.

**Scientific Question:** How does the time separation (Δt) of spacetime checks affect logical state fidelity — and how does a Clifford-only circuit compare against a T-doped circuit (Naïve vs Z-Safe T-doping)?

### Solution Overview

The solution (`task1/spacetime_sentinel.py`) builds a **4-qubit hardware-efficient ansatz** with repeating entangling layers and embeds spacetime checks using CX/CY/CZ Controlled-Pauli operators. It compares three experimental modes:

| Mode | T-Gate Placement | Effect |
|---|---|---|
| **Clifford-only** | No T-gates | Clean baseline — highest fidelity |
| **Naïve T-doped** | T inside active X-check window | Scrambles syndrome — lower fidelity |
| **Z-Safe T-doped** | T only near Z-checks (commutes) | Fidelity preserved — near Clifford |

### Key Features

- ✅ **100% guaranteed to run** — seamless built-in fallback if `qiskit-paulice` is not installed
- 🔬 Two experiments: T-doping comparison (Δt=4 fixed) + Δt sweep from 2 to 12 layers
- 📊 Two auto-generated high-resolution charts
- 🌫️ Realistic noise model: depolarising, idle decoherence, readout error

### Experiment Results

**Experiment 1 — T-Doping Mode Comparison (Δt = 4)**

| Mode | Post-Selected Fidelity | Shot Yield |
|---|---|---|
| Clifford-only | ~0.92 | ~0.78 |
| Naïve T-doped | ~0.61 | ~0.41 |
| Z-Safe T-doped | ~0.90 | ~0.76 |

**Experiment 2 — Spacetime Separation Δt Sweep**

| Δt | Clifford F | Naïve F | Z-Safe F |
|---|---|---|---|
| 2 | ~0.94 | ~0.68 | ~0.93 |
| 4 | ~0.92 | ~0.61 | ~0.90 |
| 8 | ~0.82 | ~0.49 | ~0.81 |
| 12 | ~0.74 | ~0.38 | ~0.73 |

### Scientific Finding

> **T-gates can safely coexist with spacetime codes if and only if they are placed where only Z-type checks are active.** Z and T commute (both diagonal), so the syndrome is unaffected. X-type and T do not commute, scrambling the syndrome and dropping fidelity by ~31 percentage points.

### How to Run

```bash
# Install dependencies
pip install qiskit qiskit-aer matplotlib numpy

# Run the script
cd task1
python spacetime_sentinel.py
```

Or open directly in Google Colab:
👉 **[Open Track 1 in Colab](https://colab.research.google.com/drive/1HzwvRaYCd4NOSVXM01e-Ec1RatQtl8IC?usp=sharing)**

### Output Files

| File | Description |
|---|---|
| `experiment_1_clifford_vs_tdoped.png` | Bar chart: fidelity & yield for 3 modes |
| `experiment_2_spacetime_separation.png` | Line chart: fidelity vs Δt (2 → 12) |

---

## Track 3 — Tensor Network Surface Code Decoding

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ziWxT1EatDZgkC8buoSyRost2jhmrFwM?usp=sharing)

### Problem Statement

The planar surface code is one of the most promising QEC codes for near-term hardware. Maximum-likelihood decoding is optimal but exponentially expensive if done naively. **Tensor networks** represent the joint error probability as a structured 2D network of small tensors, contracted efficiently using Matrix Product State (MPS) techniques.

**Scientific Questions:**
- **(a)** How did the contraction strategy adapt to asymmetric erasure noise where X errors are 10× more likely than Z errors?
- **(b)** How did dynamically adjusting the bond-dimension truncation threshold ε affect final logical fidelity?

### Solution Overview

The decoder (`task3/tn_decoder.py`) implements the full maximum-likelihood pipeline for the **distance-3 (and distance-5) planar surface code**:

```
Syndrome  →  GF(2) solve (e0)  →  Build 2D TN (per coset)
         →  Boundary MPS contraction + dynamic SVD truncation
         →  4-element array [P(I), P(X̄), P(Z̄), P(Ȳ)]
         →  argmax → correction C
```

### Surface Code Layout (d=3, 5×5 grid)

```
D  Z  D  Z  D      D = Data qubit      ((i+j) even)
X  D  X  D  X      Z = Z-check         (i even, j odd)  — detects X errors
D  Z  D  Z  D      X = X-check         (i odd,  j even) — detects Z errors
X  D  X  D  X
D  Z  D  Z  D
```

- **13 data qubits**, 6 Z-checks, 6 X-checks for d=3
- **Logical X̄** = X on row 0 &nbsp;|&nbsp; **Logical Z̄** = Z on column 0
- Verified with `qiskit.quantum_info.Pauli` → prints **True**

### Noise Model

| Parameter | Value | Description |
|---|---|---|
| `p_x` | 0.02 – 0.20 | X-error probability per qubit |
| `p_z` | `p_x / 10` | Z-error probability (10× biased) |
| `p_e` | 0.05 | Heralded erasure probability |
| Erased prior | 0.25 each | Uniform over I/X/Y/Z |

### Experiment Results

**Experiment 1 — TN vs Brute Force Validation**

| Metric | Result | Status |
|---|---|---|
| Max \|TN − Exact\| at ε=0 | ~1e-14 | ✅ < 1e-10 |
| Syndromes tested | 200 | ✅ |
| Qiskit algebra check | True | ✅ |
| Single X/Y/Z errors corrected (d=3) | 39/39 | ✅ |

**Experiment 2 — Fidelity vs p_x (at p_x = 0.10)**

| Decoder | Fidelity |
|---|---|
| Full (erasure + bias aware) | ~0.82 |
| Erasure-blind | ~0.76 |
| Bias-blind (symmetric p) | ~0.78 |

**Experiment 3 — SVD Threshold ε Sweep (d=3, biased noise)**

| ε | Fidelity | Mean χ | Score |
|---|---|---|---|
| 0.30 | ~0.79 | ~1.3 | Low |
| 0.10 | ~0.83 | ~1.6 | Medium |
| **0.01** | **~0.89** | **~2.1** | **Best ⭐** |
| 1e-3 | ~0.89 | ~2.9 | Good |
| 0 (exact) | ~0.90 | ~4.1 | Slower |

### Scientific Answers

**(a) Asymmetric noise and compressibility:**
Biased noise (p_x = 10 × p_z) concentrates coset probability in the I and Z̄ logical classes, making the MPS bond structure near-diagonal. At d=5, biased noise achieves mean χ ≈ 2–3 vs χ ≈ 3–4 for symmetric noise at comparable fidelity. **Noise asymmetry makes the tensor network inherently more compressible.**

**(b) Dynamic SVD threshold ε:**
Aggressive truncation (ε = 0.3) discards important correlations and costs ~5–10 fidelity points. No truncation (ε = 0) is exact but up to 5× slower. **ε ≈ 1e-2 is the sweet spot** — near-exact fidelity, 3–4× faster, highest composite score F × min(1, T_budget / t_ms).

### How to Run

```bash
# Install dependencies
pip install numpy matplotlib qiskit pytest

# Run all experiments (generates 3 figures + results.json)
cd task3
python run_experiments.py

# Run the test suite (52 tests)
pytest tests/test_decoder.py -v
```

Or open directly in Google Colab:
👉 **[Open Track 3 in Colab](https://colab.research.google.com/drive/1ziWxT1EatDZgkC8buoSyRost2jhmrFwM?usp=sharing)**

### Output Files

| File | Description |
|---|---|
| `outputs/fig1_fidelity_vs_p.png` | Exp 2: 3 decoder variants vs error rate |
| `outputs/fig2_threshold_sweep.png` | Exp 3: ε sweep for d=3 and d=5 |
| `outputs/fig3_four_element_array.png` | Exp 1: example [P(I), P(X̄), P(Z̄), P(Ȳ)] |
| `outputs/results.json` | All numerical results |

---

## Judging Criteria

| Criterion | Weight | Track 1 Coverage | Track 3 Coverage |
|---|---|---|---|
| **Final Presentation** | 40% | 2 charts + console summary table | 3 figures + results.json + Colab notebook |
| **Conceptual Breakthrough** | 30% | T-gate placement rule for spacetime codes | Noise bias → TN compressibility connection |
| **Methodology** | 20% | Post-selection filter + realistic noise model | Brute-force validation + dynamic SVD + fidelity score |
| **Qiskit Integration** | 10% | `qiskit-paulice` with fallback | `qiskit.quantum_info.Pauli` algebra check |

---

## Full Explainer

For a complete walkthrough of both problems and solutions — including step-by-step algorithm descriptions, design decision tables, and full results — see:

📄 [`docs/hackathon_solutions.md`](docs/hackathon_solutions.md)

---

## Project Structure

```
PS 1&3/
├── README.md                          ← This file
├── docs/
│   └── hackathon_solutions.md         ← Full explainer document
├── task1/
│   ├── spacetime_sentinel.py          ← Complete Track 1 Python solution
│   └── task1_solution.html            ← Track 1 web presentation
└── task3/
    ├── tn_decoder.py                  ← Core decoder library (12 functions)
    ├── run_experiments.py             ← Runs Exp 1–3, saves all outputs
    ├── Track3_Colab.ipynb             ← Colab notebook (11 cells)
    ├── README.md                      ← Track 3 standalone readme
    ├── task3_solution.html            ← Track 3 web presentation
    ├── tests/
    │   └── test_decoder.py            ← 52 pytest tests
    └── outputs/
        ├── fig1_fidelity_vs_p.png
        ├── fig2_threshold_sweep.png
        ├── fig3_four_element_array.png
        └── results.json
```

---

## Quick Start (both tracks)

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/qec-hackathon-2026.git
cd qec-hackathon-2026

# Track 1
pip install qiskit qiskit-aer matplotlib numpy
python task1/spacetime_sentinel.py

# Track 3
pip install numpy matplotlib qiskit pytest
python task3/run_experiments.py
pytest task3/tests/test_decoder.py -v
```

Or run both tracks in Google Colab with zero setup:

| Track | Colab Link |
|---|---|
| Track 1 — Spacetime Sentinel | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HzwvRaYCd4NOSVXM01e-Ec1RatQtl8IC?usp=sharing) |
| Track 3 — TN Surface Code Decoder | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ziWxT1EatDZgkC8buoSyRost2jhmrFwM?usp=sharing) |
