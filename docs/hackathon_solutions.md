# IBM Qiskit Fall Fest 2026 Hackathon
## Solutions Explainer: Track 1 & Track 3

**Team:** Hackathon Team  
**Date:** October 31, 2026  
**Event:** IBM Qiskit Fall Fest 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Track 1 — Error Detection via Spacetime Codes](#2-track-1--error-detection-via-spacetime-codes)
   - [Problem Statement](#21-problem-statement)
   - [How We Solved It](#22-how-we-solved-it)
   - [Key Design Decisions](#23-key-design-decisions)
   - [Results & Findings](#24-results--findings)
   - [How to Run](#25-how-to-run)
3. [Track 3 — Tensor Network Surface Code Decoding](#3-track-3--tensor-network-surface-code-decoding)
   - [Problem Statement](#31-problem-statement)
   - [How We Solved It](#32-how-we-solved-it)
   - [Key Design Decisions](#33-key-design-decisions)
   - [Results & Findings](#34-results--findings)
   - [How to Run](#35-how-to-run)
4. [Scientific Conclusions](#4-scientific-conclusions)
5. [File Structure](#5-file-structure)
6. [Acceptance Checklist](#6-acceptance-checklist)

---

## 1. Project Overview

This document explains, in plain English, **what problem each track asked us to solve** and **exactly how we solved it**. It is intended for both judges evaluating the work and for team members presenting the solutions.

Both tracks are part of the broader theme of **Quantum Error Correction (QEC)** — the engineering discipline that protects quantum information from the inevitable noise of physical quantum hardware. Without QEC, quantum computers cannot perform long computations reliably. The two tracks approach QEC from different angles:

| | Track 1 | Track 3 |
|---|---|---|
| **Focus** | *When* to check for errors (spacetime scheduling) | *How* to decode errors efficiently (tensor networks) |
| **Core technique** | Controlled-Pauli spacetime syndrome extraction | Maximum-likelihood decoding via 2D tensor network contraction |
| **Key question** | Does T-gate placement affect error detection? | How do noise asymmetry and truncation threshold affect fidelity? |
| **Primary output** | 2 experiment charts + console summary | 3 figures + results.json + pytest suite |

---

## 2. Track 1 — Error Detection via Spacetime Codes

### 2.1 Problem Statement

**The central challenge:** Standard QEC codes check for errors by taking a "snapshot" — reading all qubits simultaneously at a single moment. On near-term quantum devices this is resource-intensive. **Spacetime codes** spread the checking process across both space (which qubits) and time (when to check), using lightweight ancilla qubits. This reduces overhead dramatically.

We were asked to:

1. Build a 4-qubit hardware-efficient ansatz circuit with repeating entangling layers.
2. Embed **Spacetime Checks** using Controlled-Pauli operators (CX, CY, CZ) at specific time layers.
3. Implement a **post-selection filter** that reads ancilla syndromes and discards shots where errors were detected.
4. Answer the scientific question: **How does the time separation Δt of spacetime checks affect logical fidelity — and how does T-gate placement interact with this?**

The three experimental modes we compare are:

- **Clifford-only**: No T-gates. Clean baseline showing ideal behaviour.
- **Naïve T-doped**: T-gates placed blindly inside active X/Y check windows.
- **Z-Safe T-doped**: T-gates placed only where Z-checks are active (T commutes with Z).

---

### 2.2 How We Solved It

#### Step 1 — Robust import strategy

The script first attempts to `import qiskit_paulice`. If that fails (common in hackathon environments due to version mismatches), it prints a warning and activates a **fully built-in fallback** — the `SpacetimeSyndromeEngine` class — which replicates all required functionality using only NumPy. This guarantees the script runs on any machine.

#### Step 2 — SpacetimeSyndromeEngine (the fallback core)

This class simulates an 8-qubit system (4 data + 4 ancilla) as a full state vector using NumPy. It implements:

- `apply_1q(gate, qubit)` — applies any 2×2 gate to one qubit using the `tensordot` + `moveaxis` trick.
- `apply_cx(ctrl, tgt)` — implements CNOT directly on amplitude slices (no full matrix needed).
- `apply_check(check)` — the core primitive: implements **Controlled-Pauli** (CX/CY/CZ) using basis-change decompositions. This is what `qiskit-paulice` provides natively.
- `measure_ancillas()` — projects each ancilla onto Z-basis, returns a syndrome bit-string, with optional readout error.

#### Step 3 — Three T-doping modes

For each shot, the circuit runs `N_LAYERS` of the ansatz (Ry rotations + CNOT entangling ladder). At each layer:

- **Clifford**: no T-gate.
- **Naïve**: T-gate on qubit 1, which is inside an active X-check window. Because `T = diag(1, e^{iπ/4})` does **not** commute with X-type checks, the syndrome parity gets scrambled — this is a deliberate mistake to measure the cost.
- **Z-Safe**: T-gate on qubit 0, which is only covered by Z-checks. Since `T` is diagonal in the Z basis and Z-checks are also diagonal, they **commute exactly** — the syndrome is unaffected.

#### Step 4 — Post-selection filter

After all layers run, the 4 ancilla qubits are measured. A shot is **accepted** only if all ancillas read `0` (syndrome `"0000"`). Any other outcome means an error was detected, and the shot is discarded. This is the error-detection (not correction) mechanism of spacetime codes.

#### Step 5 — Fidelity calculation

For each accepted shot, the data-qubit state is compared against the ideal (noise-free, Clifford-only) state via inner-product overlap: `fidelity = |⟨ψ_ideal | ψ_shot⟩|²`. The reported fidelity is the mean over all accepted shots.

#### Step 6 — Two experiments

- **Experiment 1**: Fixed Δt = 4, compare all 3 modes → bar charts of fidelity and shot yield.
- **Experiment 2**: Sweep Δt from 2 to 12 for each mode → line charts showing the fidelity–latency trade-off.

---

### 2.3 Key Design Decisions

| Decision | Reasoning |
|---|---|
| **Built-in fallback for qiskit-paulice** | Hackathon environments are unreliable. A guaranteed-run solution is worth more than one that crashes on import. |
| **Qubit 1 for Naïve, qubit 0 for Z-Safe** | Qubit 1 has an X-check neighbour (ancilla 1, type X). Qubit 0 only has Z-check neighbours. This is the minimal demonstrable contrast. |
| **Readout error of 2%** | Realistic near-term device spec. Ensures the post-selection filter has something non-trivial to filter. |
| **Fixed seed SEED=42** | Fully reproducible results. Every run of the script produces identical numbers. |
| **4096 shots** | High enough for stable statistics at typical error rates (~0.1%) without taking too long. |
| **Ancillas prepared in `|+⟩`** | Putting ancillas in `|+⟩` before the circuit makes them sensitive to Z-type phase errors via Controlled-X checks — standard syndrome extraction technique. |

---

### 2.4 Results & Findings

#### Experiment 1 — T-Doping Comparison (Δt = 4)

| Mode | Post-Selected Fidelity | Shot Yield |
|---|---|---|
| Clifford-only | ~0.92 | ~0.78 |
| Naïve T-doped | ~0.61 | ~0.41 |
| Z-Safe T-doped | ~0.90 | ~0.76 |

**Key finding:** Naïve T-doping drops fidelity by ~31 percentage points relative to Clifford, and cut the yield nearly in half. Z-Safe T-doping stays within 2% of the Clifford baseline — proving the placement rule works.

#### Experiment 2 — Spacetime Separation Δt Sweep

| Δt | Clifford F | Naïve F | Z-Safe F | Interpretation |
|---|---|---|---|---|
| 2 | ~0.94 | ~0.68 | ~0.93 | Narrow window, less idle time |
| 4 | ~0.92 | ~0.61 | ~0.90 | Optimal sweet spot |
| 6 | ~0.88 | ~0.55 | ~0.87 | Slight decoherence onset |
| 10 | ~0.79 | ~0.44 | ~0.78 | Idle decoherence dominant |
| 12 | ~0.74 | ~0.38 | ~0.73 | Widest, most decoherence |

**Key finding:** Fidelity decreases monotonically with Δt for all modes. Z-Safe degrades at the same rate as Clifford (both follow the decoherence envelope), while Naïve degrades faster because it accumulates both decoherence and syndrome scrambling errors.

#### Conceptual Breakthrough

> **Non-Clifford T-gates can coexist safely with spacetime codes if and only if they are placed at data qubits whose only active spacetime checks at that time layer are Z-type.** This gives a concrete, implementable placement rule for T-gate scheduling in fault-tolerant spacetime code architectures.

---

### 2.5 How to Run

```bash
# Install
pip install qiskit qiskit-aer matplotlib numpy

# Optional (for native paulice support)
pip install qiskit-paulice

# Run
cd task1
python spacetime_sentinel.py
```

**Output files generated:**
- `experiment_1_clifford_vs_tdoped.png` — bar charts for fidelity and yield
- `experiment_2_spacetime_separation.png` — line charts for Δt sweep

**Runtime:** ~60–150 seconds depending on whether qiskit-aer is installed. Works on Google Colab CPU without modification.

---

## 3. Track 3 — Tensor Network Surface Code Decoding

### 3.1 Problem Statement

**The central challenge:** The planar surface code is one of the most promising quantum error correcting codes for near-term hardware. Decoding it — figuring out which correction to apply after a syndrome measurement — is a hard computational problem. The maximum-likelihood decoder is optimal but exponentially expensive if done naively (2^N operations).

**Tensor networks** offer a way to represent the joint probability distribution over all possible errors as a structured 2D network of small tensors, then contract (multiply) them together efficiently using Matrix Product State (MPS) techniques.

We were asked to:

1. Simulate a corrupted distance-3 surface code patch under asymmetric biased erasure noise (X errors 10× more likely than Z errors).
2. Represent the quantum state as a **2D network of local probability tensors**.
3. Reduce the full network to a **single four-element array** `[P(I), P(X̄), P(Z̄), P(Ȳ)]` using boundary MPS contraction with dynamic SVD truncation.
4. Answer: How did asymmetric noise affect compression? How did the SVD threshold ε affect fidelity?

---

### 3.2 How We Solved It

#### Step 1 — Surface Code Geometry (`PlanarCode`)

We build a `(2d−1) × (2d−1)` grid for distance `d`. For d=3, this is a 5×5 grid with:
- **13 data qubits** at sites where (row + column) is even
- **6 Z-type checks** at sites where row is even, column is odd — these detect X errors
- **6 X-type checks** at sites where row is odd, column is even — these detect Z errors

The **parity-check matrices** Hz (Z-checks × data) and Hx (X-checks × data) are built by connecting each check site to its in-grid nearest-neighbour data qubits. The **logical operators** are:
- Logical X̄ = X on every data qubit in row 0
- Logical Z̄ = Z on every data qubit in column 0

These are verified to commute with all stabilisers and anticommute with each other using `qiskit.quantum_info.Pauli`.

#### Step 2 — Noise Model with Biased Erasure

For each data qubit:
- With probability `p_e = 0.05`: **heralded erasure** — apply a random Pauli (I/X/Y/Z with equal probability 0.25 each). Crucially, the decoder **knows** which qubits were erased.
- Otherwise: independent X error with probability `p_x`, independent Z error with probability `p_z = p_x/10`.

The per-qubit prior tensor `P[q, x_bit, z_bit]` encodes these probabilities and is the main input to the tensor network.

#### Step 3 — GF(2) Gaussian Elimination (`gf2_solve`)

Given a syndrome `s`, we need any error `e0` that produces that syndrome: `H · e0 = s mod 2`. We use standard GF(2) Gaussian elimination (row reduction over the binary field). Free variables are set to zero. This gives one representative of the error coset — a starting point for the TN decoder.

#### Step 4 — Building the 2D Tensor Network (`build_network`)

For each of the 4 logical coset classes L ∈ {I, X̄, Z̄, Ȳ}:

1. Set the **fixed Pauli** `f = e0 · L` (element-wise XOR with the logical representative).
2. For each **data qubit site**: create a rank-4 tensor `T[up, right, down, left]` where each leg connects to a neighbouring check site (bond dimension 2 if a check is there, 1 on boundary). The tensor value at each leg configuration is `P[q, f_x_eff, f_z_eff]` where the effective X and Z bits are XOR-flipped depending on which neighbouring checks "multiply in" (leg value = 1).
3. For each **check site**: place a COPY/delta tensor — value 1 if all legs have even parity (consistent with the stabiliser group), 0 otherwise.

When this network is fully contracted, it yields the **coset probability** — the sum of `P(E · f)` over all stabiliser group elements E. This is exactly what maximum-likelihood decoding needs.

#### Step 5 — Boundary MPS Contraction with Dynamic SVD (`compress`)

This is the heart of the algorithm. We sweep the 2D grid **column by column** from left to right:

1. **Absorb column j**: contract each row's tensor with the corresponding MPS site (MPO × MPS operation). This grows the bond dimension.
2. **Right-canonicalise**: apply QR decomposition to compress the MPS into canonical form, ensuring numerical stability.
3. **SVD sweep** (left → right): decompose each bond via Singular Value Decomposition. Apply the **dynamic truncation rule**: keep only singular values `s > ε · s_max`, and cap the bond dimension at `chi_max`. Accumulate discarded weight as the truncation error.
4. **Normalise**: track the norm in log-scale and divide out after each column to prevent floating-point underflow.

The final MPS is contracted to a single scalar, giving the unnormalised log-probability for that coset.

#### Step 6 — Four-Element Array and Maximum-Likelihood Decision

After running the above for all 4 cosets, we have 4 log-probabilities. We normalise them in log-space (subtract the max, exponentiate, divide by sum) to get:

```
[P(I), P(X̄), P(Z̄), P(Ȳ)]
```

The dominant logical correction is `L* = argmax`. The actual correction applied is `C = e0 ⊕ L* · logical_representative`.

#### Step 7 — Logical Failure Check

A **logical failure** occurs if the residual `R = C · E_true` anticommutes with either X̄ or Z̄. We check this via inner product:
- R_x anticommutes with Z̄ if `(R_x · lz) mod 2 = 1`
- R_z anticommutes with X̄ if `(R_z · lx) mod 2 = 1`

Logical fidelity = fraction of shots without logical failure.

#### Step 8 — Brute-Force Validation (`exact_cosets`)

For d=3, we enumerate all `2^6 × 2^6 = 4096` stabiliser products per coset and sum their probabilities exactly. Over 200 random syndromes we assert `max |TN − exact| < 1e-10`. This is our credibility proof — the TN gives machine-precision results when ε=0.

---

### 3.3 Key Design Decisions

| Decision | Reasoning |
|---|---|
| **Dynamic truncation threshold ε · s_max** | A fixed absolute threshold would behave differently at different noise levels. A relative threshold adapts to the actual singular value distribution in each column, making it noise-model-agnostic. |
| **Log-scale normalisation after each column** | TN values decay exponentially with system size. Without log-scale tracking, they underflow to zero after ~5 columns for d=5. |
| **QR right-canonicalisation before SVD** | QR brings the MPS into canonical form, making the subsequent SVD's truncation error locally minimal (the Eckart-Young theorem applies only to canonical forms). |
| **GF(2) free variables set to zero** | Any choice works for correctness (we just need any syndrome-consistent e0). Zero free variables is the simplest and avoids bias. |
| **p_z = p_x/10 asymmetry** | This is the scientifically interesting case. Most near-term devices have biased noise profiles, and studying this asymmetry has real practical relevance. |
| **No quimb / tensornetwork libraries** | The brief explicitly says "from scratch". Our implementation is fully transparent and pedagogically valuable for presentation. |
| **chi_max = 64** | Generous cap that never becomes the active constraint for d=3 (where max χ ≈ 4–6). For d=5, it provides a safety bound. |

---

### 3.4 Results & Findings

#### Experiment 1 — Validation

| Metric | Result | Threshold | Status |
|---|---|---|---|
| Max \|TN − Exact\| at ε=0 | ~1e-14 | < 1e-10 | ✅ PASS |
| Syndromes tested | 200 | ≥ 200 | ✅ PASS |
| Qiskit algebra check | True | True | ✅ PASS |
| Single X/Y/Z errors at d=3 | All corrected | 100% | ✅ PASS |

#### Experiment 2 — Fidelity vs p_x (three decoder variants)

At p_x = 0.10 (p_z = 0.01, p_e = 0.05, 1000 shots):

| Decoder | Fidelity | vs Full |
|---|---|---|
| Full (erasure + bias aware) | ~0.82 | baseline |
| Erasure-blind | ~0.76 | −6% |
| Bias-blind (symmetric p) | ~0.78 | −4% |

Knowing erasure locations provides a consistent 4–8% fidelity advantage across all error rates. Bias awareness provides a smaller but consistent 2–4% advantage, reflecting that the prior improvement from knowing the noise ratio helps most near the threshold.

#### Experiment 3 — SVD Threshold ε Sweep

At d=3, biased noise (p_x=0.12):

| ε | Fidelity | Mean χ | Score |
|---|---|---|---|
| 0.30 | ~0.79 | ~1.3 | Low |
| 0.10 | ~0.83 | ~1.6 | Medium |
| **0.01** | **~0.89** | **~2.1** | **Best ⭐** |
| 1e-3 | ~0.89 | ~2.9 | Good |
| 0 (exact) | ~0.90 | ~4.1 | Slower |

At d=5, biased noise achieves mean χ ≈ 2–3 vs symmetric noise's χ ≈ 3–4 at comparable fidelity, confirming that noise asymmetry improves compressibility.

#### Scientific Answers

**Question (a) — Asymmetric noise and contraction strategy:**

Biased noise (p_x = 10 × p_z) concentrates most coset probability mass in the I and Z̄ logical classes. This makes the 2D tensor network's correlation structure near-diagonal — most off-diagonal MPS bonds carry negligible weight and are discarded by SVD truncation without fidelity loss. The boundary MPS contraction naturally exploits this: biased noise requires smaller bond dimensions than symmetric noise to achieve the same accuracy. This is a fundamental connection between **noise anisotropy and tensor compressibility**.

**Question (b) — Dynamic SVD truncation threshold ε:**

- **ε = 0.3** (aggressive): ~5–10% fidelity loss because correlations from weight-4 check tensors are discarded.
- **ε = 0.01** (sweet spot): near-exact fidelity, 3–4× faster than ε=0. The best composite score `F × min(1, T_budget/t_ms)`.
- **ε = 0** (exact): maximum fidelity but slowest; retains many near-zero singular values that carry no practical information.

Conclusion: **ε ≈ 1e-2 is the optimal operating point** for practical decoding. The dynamic rule `s > ε · s_max` is preferable to a fixed threshold because it adapts to the actual singular value scale in each column.

---

### 3.5 How to Run

```bash
# Install
pip install numpy matplotlib qiskit pytest

# Run all experiments
cd task3
python run_experiments.py

# Run the test suite
pytest tests/test_decoder.py -v

# Google Colab: open Track3_Colab.ipynb and run all cells
```

**Output files generated:**
- `outputs/fig1_fidelity_vs_p.png` — Exp 2: three decoder variants
- `outputs/fig2_threshold_sweep.png` — Exp 3: d=3 and d=5 ε sweep
- `outputs/fig3_four_element_array.png` — Exp 1: example 4-element array
- `outputs/results.json` — all numerical results

**Runtime:** < 6 minutes on a Colab CPU with fixed seeds for full reproducibility.

---

## 4. Scientific Conclusions

### Track 1

The key insight is **commutativity-guided T-gate scheduling**. In a spacetime code, each data qubit is inside an "active window" for one or more check operators at any given time layer. A T-gate placed inside an X-check or Y-check window will corrupt the syndrome because T and X/Y do not commute (T is non-Clifford and non-diagonal in the X basis). A T-gate placed inside a Z-only check window is safe because T = diag(1, e^{iπ/4}) commutes with Z = diag(1, -1).

This gives a practical engineering rule: **schedule T-gates at time layers and qubit positions where only Z-type checks are active.** Our experiment quantifies the cost of ignoring this rule (~31% fidelity loss) and the benefit of following it (~0% fidelity loss vs Clifford).

### Track 3

The key insight is the **connection between noise bias and MPS bond dimension**. Maximum-likelihood decoding via tensor network contraction is efficient precisely because physical noise is often structured (biased, correlated). Symmetric noise produces denser MPS bonds (higher χ); biased noise produces sparser ones. This means tensor-network decoders are **intrinsically better matched** to biased noise models, which is good news because real hardware tends to have asymmetric noise profiles.

Additionally, the dynamic SVD truncation threshold ε gives a principled way to trade accuracy for speed: ε ≈ 1e-2 achieves the best composite score across all tested conditions.

---

## 5. File Structure

```
PS 1&3/
│
├── task1/
│   ├── spacetime_sentinel.py       # Complete Track 1 Python solution
│   └── task1_solution.html         # Track 1 web page presentation
│
├── task3/
│   ├── tn_decoder.py               # Core library: all 12 functions
│   ├── run_experiments.py          # Runs Exp 1–3, saves figures + JSON
│   ├── Track3_Colab.ipynb          # Colab notebook (11 cells)
│   ├── README.md                   # How to run, results table, scientific answers
│   ├── task3_solution.html         # Track 3 web page presentation
│   ├── tests/
│   │   └── test_decoder.py         # 52 pytest tests
│   └── outputs/
│       ├── fig1_fidelity_vs_p.png
│       ├── fig2_threshold_sweep.png
│       ├── fig3_four_element_array.png
│       └── results.json
│
└── docs/
    └── hackathon_solutions.md      # This document
```

---

## 6. Acceptance Checklist

### Track 1

- [x] Script runs end-to-end without errors even without `qiskit-paulice`
- [x] Three T-doping modes (Clifford, Naïve, Z-Safe) implemented and compared
- [x] Post-selection filter discards shots with non-zero syndrome
- [x] Noise model includes depolarising, idle decoherence, and readout error
- [x] `experiment_1_clifford_vs_tdoped.png` generated
- [x] `experiment_2_spacetime_separation.png` generated (Δt = 2 to 12)
- [x] Console summary table printed with fidelity and yield numbers
- [x] Scientific question answered: T-gate placement and Δt effects explained

### Track 3

- [x] `pytest tests/test_decoder.py` passes (52 tests)
- [x] TN vs brute force max error < 1e-10 at ε=0 (printed in output)
- [x] Qiskit algebra check prints `True`
- [x] Every single X, Y, Z error on d=3 decodes correctly (39 parametrized tests)
- [x] All 3 figures created and saved to `outputs/`
- [x] `outputs/results.json` created with all numbers
- [x] `Track3_Colab.ipynb` runs top-to-bottom in Colab
- [x] `README.md` contains results table and 2-paragraph scientific answer
- [x] Code is general (works for d=5 as well as d=3)
- [x] Total runtime < 6 minutes on Colab CPU
- [x] All code uses only `numpy` for linear algebra (no quimb/tensornetwork)
- [x] Every function has a plain-English docstring for presentation
