"""
tests/test_var_calculator.py

Unit tests for VaRCalculator.

These tests verify mathematical properties that MUST hold regardless of
which simulation method is used — they are the ground truth of VaR/CVaR.

Run with:  pytest tests/test_var_calculator.py -v
"""

import pytest
import numpy as np

from src.var_calculator import VaRCalculator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CONFIDENCE = 0.99
HORIZON = 1
N_SIMS = 10_000
N_ASSETS = 3


@pytest.fixture
def calc():
    return VaRCalculator(
        confidence_level=CONFIDENCE,
        horizon_days=HORIZON,
        num_simulations=N_SIMS,
    )


@pytest.fixture
def synthetic_returns():
    """
    Simulated returns: (N_SIMS, N_ASSETS) array.
    Known to have fat tails (Student-t) so tests exercise the tail properly.
    """
    rng = np.random.default_rng(42)
    # Student-t with df=5 → fat tails, realistic for equities
    returns = rng.standard_t(df=5, size=(N_SIMS, N_ASSETS)) * 0.01
    return returns


@pytest.fixture
def prices():
    """Arbitrary last prices for 3 assets."""
    return np.array([175.0, 420.0, 140.0])


@pytest.fixture
def positions():
    return {"AAPL": 1000, "MSFT": 500, "GOOGL": 750}


# ---------------------------------------------------------------------------
# 1. PnL calculation
# ---------------------------------------------------------------------------

class TestPnLCalculation:

    def test_pnl_shape(self, calc, synthetic_returns, prices, positions):
        """PnL must be a 1D array of length N_SIMS."""
        pnl = calc.calculate_pnl(synthetic_returns, prices, positions)
        assert pnl.shape == (N_SIMS,), (
            f"Expected PnL shape ({N_SIMS},), got {pnl.shape}"
        )

    def test_pnl_no_nans(self, calc, synthetic_returns, prices, positions):
        pnl = calc.calculate_pnl(synthetic_returns, prices, positions)
        assert not np.any(np.isnan(pnl)), "PnL contains NaN values."

    def test_pnl_has_both_gains_and_losses(self, calc, synthetic_returns, prices, positions):
        """A realistic simulation must have both positive and negative PnL."""
        pnl = calc.calculate_pnl(synthetic_returns, prices, positions)
        assert np.any(pnl > 0), "No positive PnL scenarios — check simulation."
        assert np.any(pnl < 0), "No negative PnL scenarios — check simulation."

    def test_pnl_scales_with_position_size(self, calc, synthetic_returns, prices):
        """Doubling all positions must double the PnL."""
        pos1 = {"AAPL": 100, "MSFT": 100, "GOOGL": 100}
        pos2 = {"AAPL": 200, "MSFT": 200, "GOOGL": 200}

        pnl1 = calc.calculate_pnl(synthetic_returns, prices, pos1)
        pnl2 = calc.calculate_pnl(synthetic_returns, prices, pos2)

        np.testing.assert_allclose(
            pnl2, pnl1 * 2, rtol=1e-9,
            err_msg="PnL did not scale linearly with position size."
        )


# ---------------------------------------------------------------------------
# 2. VaR/CVaR mathematical properties
# ---------------------------------------------------------------------------

class TestVaRProperties:

    def test_cvar_always_geq_var(self, calc, synthetic_returns, prices, positions):
        """
        CVaR ≥ VaR in absolute terms, by definition.
        This must hold for ALL confidence levels and ALL distributions.
        """
        pnl = calc.calculate_pnl(synthetic_returns, prices, positions)
        var_values, cvar_values = calc.calculate_var_cvar(pnl)

        # Both are losses (negative), so |CVaR| >= |VaR|
        if isinstance(var_values, dict):
            for key in var_values:
                assert abs(cvar_values[key]) >= abs(var_values[key]), (
                    f"CVaR < VaR for key '{key}': "
                    f"CVaR={cvar_values[key]:.2f}, VaR={var_values[key]:.2f}"
                )
        else:
            assert abs(cvar_values) >= abs(var_values), (
                f"CVaR ({cvar_values:.2f}) < VaR ({var_values:.2f}). "
                f"This violates the definition of CVaR."
            )

    def test_var_is_negative(self, calc, synthetic_returns, prices, positions):
        """VaR should be a loss (negative number) for a long portfolio."""
        pnl = calc.calculate_pnl(synthetic_returns, prices, positions)
        var_values, _ = calc.calculate_var_cvar(pnl)

        var = var_values if not isinstance(var_values, dict) else list(var_values.values())[0]
        assert var < 0, (
            f"VaR should be negative (a loss), got {var:.2f}. "
            f"Check the sign convention in VaRCalculator."
        )

    def test_cvar_is_negative(self, calc, synthetic_returns, prices, positions):
        """CVaR should also be negative (it's the average of the worst losses)."""
        pnl = calc.calculate_pnl(synthetic_returns, prices, positions)
        _, cvar_values = calc.calculate_var_cvar(pnl)

        cvar = cvar_values if not isinstance(cvar_values, dict) else list(cvar_values.values())[0]
        assert cvar < 0, (
            f"CVaR should be negative (a loss), got {cvar:.2f}."
        )

    def test_higher_confidence_gives_larger_var(self, prices, positions, synthetic_returns):
        """
        A 99% VaR must be >= 95% VaR in absolute terms.
        More confidence = further into the tail = bigger loss.
        """
        calc_95 = VaRCalculator(0.95, HORIZON, N_SIMS)
        calc_99 = VaRCalculator(0.99, HORIZON, N_SIMS)

        pnl = calc_95.calculate_pnl(synthetic_returns, prices, positions)

        var_95, _ = calc_95.calculate_var_cvar(pnl)
        var_99, _ = calc_99.calculate_var_cvar(pnl)

        # Extract scalar if dict
        if isinstance(var_95, dict):
            var_95 = list(var_95.values())[0]
            var_99 = list(var_99.values())[0]

        assert abs(var_99) >= abs(var_95), (
            f"99% VaR ({var_99:.2f}) is not >= 95% VaR ({var_95:.2f}). "
            f"Higher confidence must yield larger loss threshold."
        )

    def test_var_percentile_consistency(self, calc, prices, positions):
        """
        For a known distribution, VaR should match the empirical percentile.

        We create a PnL with known structure and verify the calculator
        returns a value consistent with np.percentile.
        """
        rng = np.random.default_rng(0)
        # PnL in $, symmetric normal, mean=0, std=$1000
        known_pnl = rng.normal(0, 1000, 100_000)

        var, cvar = calc.calculate_var_cvar(known_pnl)

        # Expected VaR at 99%: the 1st percentile of PnL
        expected_var = np.percentile(known_pnl, (1 - CONFIDENCE) * 100)

        var_scalar = var if not isinstance(var, dict) else list(var.values())[0]

        np.testing.assert_allclose(
            var_scalar, expected_var, rtol=0.05,
            err_msg=(
                f"VaR ({var_scalar:.2f}) deviates >5% from expected "
                f"percentile ({expected_var:.2f})."
            )
        )

    def test_cvar_is_mean_of_tail(self, calc, prices, positions):
        """
        CVaR must equal the mean of losses beyond VaR.
        This is the analytical definition.
        """
        rng = np.random.default_rng(0)
        known_pnl = rng.normal(0, 1000, 100_000)

        var, cvar = calc.calculate_var_cvar(known_pnl)

        var_scalar = var if not isinstance(var, dict) else list(var.values())[0]
        cvar_scalar = cvar if not isinstance(cvar, dict) else list(cvar.values())[0]

        # Expected CVaR: mean of all scenarios worse than VaR
        tail = known_pnl[known_pnl <= var_scalar]
        expected_cvar = np.mean(tail)

        np.testing.assert_allclose(
            cvar_scalar, expected_cvar, rtol=0.05,
            err_msg=(
                f"CVaR ({cvar_scalar:.2f}) does not match mean of tail "
                f"({expected_cvar:.2f})."
            )
        )


# ---------------------------------------------------------------------------
# 3. Horizon scaling
# ---------------------------------------------------------------------------

class TestHorizonScaling:

    def test_multi_day_var_larger_than_one_day(self, synthetic_returns, prices, positions):
        """
        10-day VaR should be larger in absolute terms than 1-day VaR.
        Under sqrt(T) scaling: VaR_10 = VaR_1 * sqrt(10).
        """
        calc_1d = VaRCalculator(CONFIDENCE, horizon_days=1, num_simulations=N_SIMS)
        calc_10d = VaRCalculator(CONFIDENCE, horizon_days=10, num_simulations=N_SIMS)

        pnl = calc_1d.calculate_pnl(synthetic_returns, prices, positions)

        var_1d, _ = calc_1d.calculate_var_cvar(pnl)
        var_10d, _ = calc_10d.calculate_var_cvar(pnl)

        v1 = var_1d if not isinstance(var_1d, dict) else list(var_1d.values())[0]
        v10 = var_10d if not isinstance(var_10d, dict) else list(var_10d.values())[0]

        assert abs(v10) > abs(v1), (
            f"10-day VaR ({v10:.2f}) should be larger than 1-day VaR ({v1:.2f})."
        )
