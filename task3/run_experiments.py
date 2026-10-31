"""
run_experiments.py
==================
IBM Qiskit Fall Fest 2026 Hackathon – Track 3
"Tensor Network Surface Code Decoding & Reduction"

Runs Experiments 1–3, prints a clean console summary,
and saves all figures + results.json to outputs/.

Usage:
    python run_experiments.py

Runtime target: < 6 minutes on a Colab CPU.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tn_decoder import (
    PlanarCode, NoiseParams, priors, sample_error,
    build_network, compress, contract, decode_tn,
    exact_cosets, gf2_solve, run, fidelity_score,
    qiskit_check, LOGICAL_NAMES, RunResult,
)

# ── output directory ─────────────────────────────────────────────────────────
OUTDIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTDIR, exist_ok=True)

# ── fixed RNG seeds (reproducible) ──────────────────────────────────────────
SEED_EXP1 = 0
SEED_EXP2 = 1
SEED_EXP3 = 2

# ── colour palette ───────────────────────────────────────────────────────────
C_FULL     = "#2196F3"   # full decoder
C_BLIND    = "#F44336"   # erasure-blind
C_BBLIND   = "#FF9800"   # bias-blind
C_BIASED   = "#4CAF50"   # biased noise
C_SYM      = "#9C27B0"   # symmetric noise


# ─────────────────────────────────────────────────────────────────────────────
# EXP 1 – VALIDATION: TN vs BRUTE FORCE
# ─────────────────────────────────────────────────────────────────────────────

def exp1_validation() -> Dict:
    """
    Exp 1: Compare TN decoder against brute-force exact reference.

    Over ≥200 random syndromes (d=3, moderate noise, no erasure):
      - Run both exact_cosets and contract(eps=0).
      - Report max |TN − exact| (must be < 1e-10 at eps=0).
      - Print one example 4-element array.
    """
    print("\n" + "═"*60)
    print("  EXP 1 – TN vs BRUTE FORCE VALIDATION")
    print("═"*60)

    code   = PlanarCode(d=3)
    params = NoiseParams(p_x=0.08, p_z=0.008, p_e=0.0)
    rng    = np.random.default_rng(SEED_EXP1)
    N_VAL  = 200

    max_err    = 0.0
    example    = None
    example_exact = None

    for trial in range(N_VAL):
        E_x, E_z, erased = sample_error(code, params, rng)
        sx = (code.Hz @ E_x) % 2
        sz = (code.Hx @ E_z) % 2
        P  = priors(code.n_data, params)

        e0_x = gf2_solve(code.Hz, sx)
        e0_z = gf2_solve(code.Hx, sz)

        # Exact reference
        exact = exact_cosets(code, P, e0_x, e0_z)

        # TN with eps=0 (no truncation → exact)
        tn, _, _, _ = contract(code, P, e0_x, e0_z, eps=0.0, chi_max=256)

        err = float(np.max(np.abs(tn - exact)))
        max_err = max(max_err, err)

        if trial == 0:
            example       = tn.copy()
            example_exact = exact.copy()

    verdict = "PASS ✓" if max_err < 1e-10 else "FAIL ✗"
    print(f"\n  Max |TN − Exact| over {N_VAL} syndromes: {max_err:.2e}  [{verdict}]")
    print(f"\n  Example 4-element array (trial 0):")
    for i, name in enumerate(LOGICAL_NAMES):
        bar = "█" * int(example[i] * 40)
        print(f"    P({name:2s}) = {example[i]:.8f}  {bar}")
    print(f"\n  Dominant correction: {LOGICAL_NAMES[int(np.argmax(example))]}")

    # ── figure: bar chart of example 4-element array ─────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    colours = [C_FULL, C_BLIND, C_BBLIND, C_BIASED]
    bars = ax.bar(LOGICAL_NAMES, example, color=colours,
                  edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, example):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Probability", fontsize=11)
    ax.set_xlabel("Logical Class", fontsize=11)
    ax.set_title(
        f"Four-Element Array [P(I), P(X̄), P(Z̄), P(Ȳ)]  ·  d=3, p_x=0.08\n"
        f"Max |TN − Exact| = {max_err:.2e}  [{verdict}]",
        fontsize=10
    )
    fig.tight_layout()
    out1 = os.path.join(OUTDIR, "fig3_four_element_array.png")
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [SAVED] {out1}")

    return {
        "max_error_tn_vs_exact": float(max_err),
        "verdict": verdict,
        "example_array": example.tolist(),
        "example_exact": example_exact.tolist(),
        "n_trials": N_VAL,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EXP 2 – FIDELITY VS p_x  (three decoders)
# ─────────────────────────────────────────────────────────────────────────────

def exp2_fidelity_vs_px() -> Dict:
    """
    Exp 2: Fidelity vs p_x for three decoder variants.

    d=3, p_z = p_x/10, p_e=0.05, p_x ∈ {0.02, 0.04, ..., 0.20}, 1000 shots.

    Three decoders:
      (i)  Full decoder          – uses erasure flags + bias
      (ii) Erasure-blind decoder – ignores erasure locations
      (iii)Bias-blind decoder    – uses symmetric p = (p_x+p_z)/2
    """
    print("\n" + "═"*60)
    print("  EXP 2 – FIDELITY vs p_x  (three decoder variants)")
    print("═"*60)

    code  = PlanarCode(d=3)
    px_vals = [round(v, 3) for v in np.arange(0.02, 0.21, 0.02)]
    N_SHOTS = 1000
    EPS     = 1e-2
    CHI_MAX = 32

    results = {
        "px":          px_vals,
        "full":        {"fidelity": [], "se": [], "score": []},
        "erasure_blind": {"fidelity": [], "se": [], "score": []},
        "bias_blind":  {"fidelity": [], "se": [], "score": []},
    }

    print(f"\n  {'p_x':>6} {'Full F':>8} {'Blind-E F':>10} {'Blind-B F':>10}")
    print("  " + "-"*38)

    for px in px_vals:
        params = NoiseParams(p_x=px, p_z=px/10, p_e=0.05)
        seed_i = SEED_EXP2 + int(px * 1000)

        r_full  = run(code, params, N_SHOTS, eps=EPS, chi_max=CHI_MAX,
                      seed=seed_i)
        r_blind = run(code, params, N_SHOTS, eps=EPS, chi_max=CHI_MAX,
                      seed=seed_i, erasure_blind=True)
        r_bblind= run(code, params, N_SHOTS, eps=EPS, chi_max=CHI_MAX,
                      seed=seed_i, bias_blind=True)

        for key, r in [("full", r_full),
                        ("erasure_blind", r_blind),
                        ("bias_blind", r_bblind)]:
            results[key]["fidelity"].append(r.fidelity)
            results[key]["se"].append(r.fidelity_se)
            results[key]["score"].append(r.score)

        print(f"  {px:>6.3f} {r_full.fidelity:>8.4f} "
              f"{r_blind.fidelity:>10.4f} {r_bblind.fidelity:>10.4f}")

    # ── figure ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    configs = [
        ("full",         C_FULL,   "Full decoder (erasure + bias)",   "o-"),
        ("erasure_blind",C_BLIND,  "Erasure-blind decoder",           "s--"),
        ("bias_blind",   C_BBLIND, "Bias-blind decoder (symmetric p)","^:"),
    ]
    for key, colour, label, fmt in configs:
        ax.errorbar(
            px_vals,
            results[key]["fidelity"],
            yerr=[1.96*se for se in results[key]["se"]],
            fmt=fmt, color=colour, label=label,
            linewidth=2, markersize=6, capsize=4
        )

    ax.set_xlabel("Bit-flip error rate p_x  (p_z = p_x / 10)", fontsize=11)
    ax.set_ylabel("Logical Fidelity", fontsize=11)
    ax.set_title(
        "Exp 2 – Fidelity vs p_x: Erasure & Bias Awareness\n"
        "d=3, 1000 shots, p_e=0.05, ε=1e-2",
        fontsize=11
    )
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(px_vals)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    out2 = os.path.join(OUTDIR, "fig1_fidelity_vs_p.png")
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [SAVED] {out2}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# EXP 3 – THRESHOLD SWEEP  (ε vs fidelity, d=3 and d=5)
# ─────────────────────────────────────────────────────────────────────────────

def exp3_threshold_sweep() -> Dict:
    """
    Exp 3: SVD truncation threshold ε sweep.

    ε ∈ {0.3, 0.1, 0.03, 0.01, 1e-3, 1e-5, 0}
    biased (p_x=0.12, p_z=0.012) vs symmetric (p_x=p_z=0.066)
    d=3: 1000 shots, d=5: 300 shots.
    """
    print("\n" + "═"*60)
    print("  EXP 3 – THRESHOLD ε SWEEP  (d=3 and d=5)")
    print("═"*60)

    eps_vals   = [0.3, 0.1, 0.03, 0.01, 1e-3, 1e-5, 0.0]
    eps_labels = ["0.3","0.1","0.03","0.01","1e-3","1e-5","0"]

    configs = [
        (PlanarCode(3), NoiseParams(p_x=0.12, p_z=0.012), 1000, "d=3 biased",    C_BIASED),
        (PlanarCode(3), NoiseParams(p_x=0.066,p_z=0.066), 1000, "d=3 symmetric", C_SYM),
        (PlanarCode(5), NoiseParams(p_x=0.12, p_z=0.012),  300, "d=5 biased",    "#00BCD4"),
        (PlanarCode(5), NoiseParams(p_x=0.066,p_z=0.066),  300, "d=5 symmetric", "#FF5722"),
    ]

    results_eps = {}
    for code, params, n_shots, label, _ in configs:
        results_eps[label] = {
            "eps": eps_vals, "eps_labels": eps_labels,
            "fidelity":[], "se":[], "chi":[], "trunc_err":[],
            "time_ms":[], "score":[]
        }

    for code, params, n_shots, label, _ in configs:
        print(f"\n  {label}  ({n_shots} shots each)")
        print(f"  {'ε':>6} {'F':>7} {'χ':>6} {'trunc':>8} {'t_ms':>8} {'score':>7}")
        print("  " + "-"*50)

        for i_e, eps in enumerate(eps_vals):
            r = run(code, params, n_shots,
                    eps=eps, chi_max=64,
                    seed=SEED_EXP3 + int(eps*1e5) + code.d*100)

            d = results_eps[label]
            d["fidelity"].append(r.fidelity)
            d["se"].append(r.fidelity_se)
            d["chi"].append(r.mean_chi)
            d["trunc_err"].append(r.mean_trunc_error)
            d["time_ms"].append(r.mean_time_ms)
            d["score"].append(r.score)

            print(f"  {eps_labels[i_e]:>6} {r.fidelity:>7.4f} "
                  f"{r.mean_chi:>6.1f} {r.mean_trunc_error:>8.4f} "
                  f"{r.mean_time_ms:>8.3f} {r.score:>7.4f}")

    # ── figure: 2-panel (d=3 left, d=5 right) ────────────────────────
    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 2, figure=fig)

    for panel_idx, d_str in enumerate(["d=3", "d=5"]):
        ax1 = fig.add_subplot(gs[panel_idx])
        ax2 = ax1.twinx()
        ax2.set_ylabel("Mean bond dimension χ", fontsize=10, color="#555")
        ax2.tick_params(axis="y", labelcolor="#555")

        for code, params, n_shots, label, colour in configs:
            if not label.startswith(d_str):
                continue
            d = results_eps[label]
            x = np.arange(len(eps_vals))
            ls = "-" if "biased" in label else "--"
            lab_noise = "Biased" if "biased" in label else "Symmetric"

            ax1.plot(x, d["fidelity"], ls+"o", color=colour,
                     label=f"F – {lab_noise}", linewidth=2, markersize=5)
            ax2.plot(x, d["chi"], ls+"s", color=colour, alpha=0.45,
                     label=f"χ – {lab_noise}", linewidth=1.5, markersize=4)

        ax1.set_xticks(np.arange(len(eps_vals)))
        ax1.set_xticklabels(eps_labels, fontsize=9)
        ax1.set_xlabel("SVD threshold ε", fontsize=11)
        ax1.set_ylabel("Logical Fidelity", fontsize=11)
        ax1.set_title(
            f"Exp 3 – Threshold Sweep  ·  {d_str}\n"
            f"Solid = fidelity  |  Dashed = bond dimension χ",
            fontsize=10
        )
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.3)
        lines1, labs1 = ax1.get_legend_handles_labels()
        ax1.legend(lines1, labs1, fontsize=8, loc="lower left")

    fig.suptitle(
        "Exp 3 – SVD Truncation Threshold vs Fidelity & Bond Dimension\n"
        "Biased (p_x=0.12, p_z=0.012) vs Symmetric (p_x=p_z=0.066)",
        fontsize=11, y=1.02
    )
    fig.tight_layout()

    out3 = os.path.join(OUTDIR, "fig2_threshold_sweep.png")
    fig.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [SAVED] {out3}")

    return results_eps


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_total = time.perf_counter()

    print("\n" + "█"*60)
    print("  TENSOR NETWORK SURFACE CODE DECODER")
    print("  IBM Qiskit Fall Fest 2026 – Track 3")
    print("█"*60)

    # ── Qiskit algebra check ──────────────────────────────────────────
    code3 = PlanarCode(d=3)
    qiskit_ok = qiskit_check(code3)

    # ── Run experiments ───────────────────────────────────────────────
    r1 = exp1_validation()
    r2 = exp2_fidelity_vs_px()
    r3 = exp3_threshold_sweep()

    # ── Save results.json ─────────────────────────────────────────────
    results_json = {
        "exp1_validation":       r1,
        "exp2_fidelity_vs_px":   r2,
        "exp3_threshold_sweep":  r3,
        "qiskit_check_passed":   qiskit_ok,
    }
    json_path = os.path.join(OUTDIR, "results.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\n[SAVED] {json_path}")

    elapsed = time.perf_counter() - t_total
    print(f"\n[DONE] Total runtime: {elapsed/60:.1f} minutes")

    # ── Final console summary ─────────────────────────────────────────
    print("\n" + "═"*60)
    print("  FINAL SUMMARY")
    print("═"*60)
    print(f"\n  Exp 1 – TN vs Brute Force")
    print(f"    Max |TN − Exact|: {r1['max_error_tn_vs_exact']:.2e}  [{r1['verdict']}]")
    print(f"    Example array:    {[f'{v:.4f}' for v in r1['example_array']]}")

    print(f"\n  Exp 2 – Fidelity vs p_x  (at p_x = 0.10)")
    idx10 = [round(v,3) for v in np.arange(0.02,0.21,0.02)].index(0.10)
    print(f"    Full decoder:      {r2['full']['fidelity'][idx10]:.4f}")
    print(f"    Erasure-blind:     {r2['erasure_blind']['fidelity'][idx10]:.4f}")
    print(f"    Bias-blind:        {r2['bias_blind']['fidelity'][idx10]:.4f}")

    print(f"\n  Exp 3 – Threshold sweep (d=3, biased, best ε)")
    d3b = r3["d=3 biased"]
    best_i = int(np.argmax(d3b["fidelity"]))
    eps_labels = ["0.3","0.1","0.03","0.01","1e-3","1e-5","0"]
    print(f"    Best ε = {eps_labels[best_i]:5s}  →  F = {d3b['fidelity'][best_i]:.4f}"
          f"  χ = {d3b['chi'][best_i]:.1f}")

    print(f"\n  Qiskit algebra check: {qiskit_ok}")
    print("\n  Output files:")
    for fname in ["fig1_fidelity_vs_p.png", "fig2_threshold_sweep.png",
                  "fig3_four_element_array.png", "results.json"]:
        print(f"    outputs/{fname}")
    print("═"*60)


if __name__ == "__main__":
    main()
