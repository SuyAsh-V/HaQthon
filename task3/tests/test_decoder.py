"""
tests/test_decoder.py
=====================
Pytest suite for the Track-3 tensor-network decoder.

Tests:
  1. Code validity – correct qubit / check counts, Hz/Hx shapes
  2. GF(2) solver – basic correctness + all-zero syndrome
  3. TN == brute force at eps=0  (over 10 random syndromes)
  4. Single X, Y, Z error always corrected for d=3, eps=0

Run:
    cd task3
    pytest tests/test_decoder.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from tn_decoder import (
    PlanarCode, NoiseParams, priors, sample_error,
    gf2_solve, contract, exact_cosets, decode_tn,
    check_logical_failure, LOGICAL_NAMES,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def code3():
    return PlanarCode(d=3)

@pytest.fixture(scope="module")
def code5():
    return PlanarCode(d=5)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 – Code validity
# ─────────────────────────────────────────────────────────────────────────────

class TestCodeValidity:
    def test_d3_qubit_counts(self, code3):
        """d=3 patch must have exactly 13 data qubits, 6 Z-checks, 6 X-checks."""
        assert code3.n_data    == 13, f"Expected 13 data qubits, got {code3.n_data}"
        assert code3.n_z_check == 6,  f"Expected 6 Z-checks, got {code3.n_z_check}"
        assert code3.n_x_check == 6,  f"Expected 6 X-checks, got {code3.n_x_check}"

    def test_d5_qubit_counts(self, code5):
        """d=5 patch must have exactly 41 data qubits, 20 Z-checks, 20 X-checks."""
        # 5×5 code on 9×9 grid: 41 data, 20 Z, 20 X
        assert code5.n_data    == 41
        assert code5.n_z_check == 20
        assert code5.n_x_check == 20

    def test_parity_matrix_shapes(self, code3):
        """Hz and Hx must have correct shapes."""
        assert code3.Hz.shape == (6,  13)
        assert code3.Hx.shape == (6,  13)

    def test_parity_matrix_binary(self, code3):
        """Hz and Hx must contain only 0 and 1."""
        assert set(np.unique(code3.Hz)).issubset({0, 1})
        assert set(np.unique(code3.Hx)).issubset({0, 1})

    def test_logical_x_in_row0(self, code3):
        """Logical X̄ must hit exactly the data qubits in row 0."""
        row0_qubits = {i for i,(r,c) in enumerate(code3.data_sites) if r == 0}
        lx_support  = set(np.where(code3.lx)[0])
        assert lx_support == row0_qubits

    def test_logical_z_in_col0(self, code3):
        """Logical Z̄ must hit exactly the data qubits in column 0."""
        col0_qubits = {i for i,(r,c) in enumerate(code3.data_sites) if c == 0}
        lz_support  = set(np.where(code3.lz)[0])
        assert lz_support == col0_qubits

    def test_logicals_commute_with_stabilisers(self, code3):
        """X̄ and Z̄ must each commute with all stabilisers."""
        # X̄ (X-type) commutes with Z-stabilisers (Hz): inner product = 0 for each row
        for row in code3.Hz:
            assert int(np.dot(code3.lx, row)) % 2 == 0, \
                "Logical X̄ anticommutes with a Z-stabiliser!"
        # Z̄ (Z-type) commutes with X-stabilisers (Hx)
        for row in code3.Hx:
            assert int(np.dot(code3.lz, row)) % 2 == 0, \
                "Logical Z̄ anticommutes with an X-stabiliser!"

    def test_logicals_anticommute(self, code3):
        """X̄ and Z̄ must anticommute: lx · lz = 1 (mod 2)."""
        assert int(np.dot(code3.lx, code3.lz)) % 2 == 1, \
            "X̄ and Z̄ should anticommute!"

    def test_stabilisers_commute(self, code3):
        """All Z-stabilisers must commute with all X-stabilisers."""
        for z_row in code3.Hz:
            for x_row in code3.Hx:
                # Z commutes with X iff they share an even number of sites
                assert int(np.dot(z_row, x_row)) % 2 == 0, \
                    "A Z-stabiliser anticommutes with an X-stabiliser!"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 – GF(2) Solver
# ─────────────────────────────────────────────────────────────────────────────

class TestGF2Solve:
    def test_zero_syndrome(self, code3):
        """Zero syndrome → zero solution."""
        s = np.zeros(code3.n_z_check, dtype=np.uint8)
        e = gf2_solve(code3.Hz, s)
        assert np.array_equal(e, np.zeros(code3.n_data, dtype=np.uint8))

    def test_single_error_recovery(self, code3):
        """Single X error on qubit 0 → solution satisfies Hz·e = sx."""
        e_true = np.zeros(code3.n_data, dtype=np.uint8)
        e_true[0] = 1
        sx = (code3.Hz @ e_true) % 2
        e0 = gf2_solve(code3.Hz, sx)
        assert np.array_equal((code3.Hz @ e0) % 2, sx), \
            "GF(2) solver did not produce a valid solution!"

    def test_multiple_errors(self, code3):
        """Two X errors: solver must return a consistent solution."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            e = rng.integers(0, 2, size=code3.n_data, dtype=np.uint8)
            sx = (code3.Hz @ e) % 2
            e0 = gf2_solve(code3.Hz, sx)
            assert np.array_equal((code3.Hz @ e0) % 2, sx), \
                "GF(2) solver inconsistency!"

    def test_z_solver(self, code3):
        """Hx solver test for Z errors."""
        rng = np.random.default_rng(99)
        for _ in range(20):
            e = rng.integers(0, 2, size=code3.n_data, dtype=np.uint8)
            sz = (code3.Hx @ e) % 2
            e0 = gf2_solve(code3.Hx, sz)
            assert np.array_equal((code3.Hx @ e0) % 2, sz)


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 – TN == Brute Force  (eps=0)
# ─────────────────────────────────────────────────────────────────────────────

class TestTNvsExact:
    def test_max_error_under_threshold(self, code3):
        """
        Max |TN − exact| < 1e-10 at eps=0, over 10 random syndromes.
        (The full 200-syndrome check is run in run_experiments.py; here
        we use 10 for test-suite speed.)
        """
        params = NoiseParams(p_x=0.08, p_z=0.008, p_e=0.0)
        rng    = np.random.default_rng(1234)
        max_err = 0.0

        for _ in range(10):
            E_x, E_z, _ = sample_error(code3, params, rng)
            sx = (code3.Hz @ E_x) % 2
            sz = (code3.Hx @ E_z) % 2
            P  = priors(code3.n_data, params)
            e0x = gf2_solve(code3.Hz, sx)
            e0z = gf2_solve(code3.Hx, sz)

            exact = exact_cosets(code3, P, e0x, e0z)
            tn, _, _, _ = contract(code3, P, e0x, e0z, eps=0.0, chi_max=256)

            err = float(np.max(np.abs(tn - exact)))
            max_err = max(max_err, err)

        assert max_err < 1e-10, \
            f"TN vs brute-force max error {max_err:.2e} exceeds 1e-10!"

    def test_probabilities_sum_to_one(self, code3):
        """4-element array must be a valid probability distribution."""
        params = NoiseParams(p_x=0.05, p_z=0.005, p_e=0.0)
        rng    = np.random.default_rng(5678)
        E_x, E_z, _ = sample_error(code3, params, rng)
        sx = (code3.Hz @ E_x) % 2
        sz = (code3.Hx @ E_z) % 2
        P  = priors(code3.n_data, params)
        e0x = gf2_solve(code3.Hz, sx)
        e0z = gf2_solve(code3.Hx, sz)
        tn, _, _, _ = contract(code3, P, e0x, e0z, eps=1e-2, chi_max=32)
        assert abs(tn.sum() - 1.0) < 1e-9, \
            f"4-element array sums to {tn.sum():.8f}, not 1!"
        assert np.all(tn >= 0), "Negative probability in 4-element array!"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 – Single-qubit errors always corrected  (d=3, eps=0)
# ─────────────────────────────────────────────────────────────────────────────

class TestSingleQubitCorrection:
    """
    Every single-qubit X, Y, or Z error on a d=3 patch (no erasure)
    must be decoded correctly at eps=0.
    """

    @pytest.mark.parametrize("qubit", range(13))
    def test_single_x_corrected(self, code3, qubit):
        E_x = np.zeros(code3.n_data, dtype=np.uint8); E_x[qubit] = 1
        E_z = np.zeros(code3.n_data, dtype=np.uint8)
        params = NoiseParams(p_x=0.05, p_z=0.005, p_e=0.0)
        P = priors(code3.n_data, params)
        sx = (code3.Hz @ E_x) % 2
        sz = (code3.Hx @ E_z) % 2
        C_x, C_z, _, _, _, _ = decode_tn(code3, sx, sz, P, eps=0.0, chi_max=256)
        failed = check_logical_failure(code3, C_x, C_z, E_x, E_z)
        assert not failed, f"Single X error on qubit {qubit} not corrected!"

    @pytest.mark.parametrize("qubit", range(13))
    def test_single_z_corrected(self, code3, qubit):
        E_x = np.zeros(code3.n_data, dtype=np.uint8)
        E_z = np.zeros(code3.n_data, dtype=np.uint8); E_z[qubit] = 1
        params = NoiseParams(p_x=0.05, p_z=0.005, p_e=0.0)
        P = priors(code3.n_data, params)
        sx = (code3.Hz @ E_x) % 2
        sz = (code3.Hx @ E_z) % 2
        C_x, C_z, _, _, _, _ = decode_tn(code3, sx, sz, P, eps=0.0, chi_max=256)
        failed = check_logical_failure(code3, C_x, C_z, E_x, E_z)
        assert not failed, f"Single Z error on qubit {qubit} not corrected!"

    @pytest.mark.parametrize("qubit", range(13))
    def test_single_y_corrected(self, code3, qubit):
        E_x = np.zeros(code3.n_data, dtype=np.uint8); E_x[qubit] = 1
        E_z = np.zeros(code3.n_data, dtype=np.uint8); E_z[qubit] = 1
        params = NoiseParams(p_x=0.05, p_z=0.005, p_e=0.0)
        P = priors(code3.n_data, params)
        sx = (code3.Hz @ E_x) % 2
        sz = (code3.Hx @ E_z) % 2
        C_x, C_z, _, _, _, _ = decode_tn(code3, sx, sz, P, eps=0.0, chi_max=256)
        failed = check_logical_failure(code3, C_x, C_z, E_x, E_z)
        assert not failed, f"Single Y error on qubit {qubit} not corrected!"
