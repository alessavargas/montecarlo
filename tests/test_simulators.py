"""
tests/test_simulators.py

Unit tests for the 4 Monte Carlo simulation methods.

Run with:  pytest tests/test_simulators.py -v
"""

import pytest
import numpy as np
import pandas as pd

from src.simulators import (
    CholeskyNormal,
    CholeskyEmpirical,
    PCANormal,
    PCAEmpirical,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_ASSETS = 3
N_OBS = 252  # 1 trading year
N_SIMS = 5_000
RANDOM_SEED = 42


@pytest.fixture
def sample_returns() -> pd.DataFrame:
    """Synthetic daily log-returns for 3 correlated assets."""
    rng = np.random.default_rng(RANDOM_SEED)

    # Build a realistic covariance structure
    vols = np.array([0.02, 0.018, 0.022])          # daily vol ~1-2%
    corr = np.array([
        [1.00, 0.65, 0.55],
        [0.65, 1.00, 0.50],
        [0.55, 0.50, 1.00],
    ])
    cov = np.diag(vols) @ corr @ np.diag(vols)
    L = np.linalg.cholesky(cov)

    raw = rng.standard_normal((N_OBS, N_ASSETS))
    returns_array = raw @ L.T

    return pd.DataFrame(
        returns_array,
        columns=["AAPL", "MSFT", "GOOGL"],
    )


@pytest.fixture
def all_simulators():
    return {
        "CholeskyNormal": CholeskyNormal(),
        "CholeskyEmpirical": CholeskyEmpirical(),
        "PCANormal": PCANormal(),
        "PCAEmpirical": PCAEmpirical(),
    }


# ---------------------------------------------------------------------------
# 1. Output shape
# ---------------------------------------------------------------------------

class TestOutputShape:
    """All simulators must return (N_SIMS, N_ASSETS) arrays."""

    @pytest.mark.parametrize("SimClass", [
        CholeskyNormal, CholeskyEmpirical, PCANormal, PCAEmpirical
    ])
    def test_shape(self, SimClass, sample_returns):
        sim = SimClass()
        result = sim.simulate(sample_returns, N_SIMS)
        assert result.shape == (N_SIMS, N_ASSETS), (
            f"{SimClass.__name__} returned shape {result.shape}, "
            f"expected ({N_SIMS}, {N_ASSETS})"
        )

    @pytest.mark.parametrize("SimClass", [
        CholeskyNormal, CholeskyEmpirical, PCANormal, PCAEmpirical
    ])
    def test_no_nans(self, SimClass, sample_returns):
        sim = SimClass()
        result = sim.simulate(sample_returns, N_SIMS)
        assert not np.any(np.isnan(result)), (
            f"{SimClass.__name__} produced NaN values in simulated returns."
        )

    @pytest.mark.parametrize("SimClass", [
        CholeskyNormal, CholeskyEmpirical, PCANormal, PCAEmpirical
    ])
    def test_no_infs(self, SimClass, sample_returns):
        sim = SimClass()
        result = sim.simulate(sample_returns, N_SIMS)
        assert not np.any(np.isinf(result)), (
            f"{SimClass.__name__} produced infinite values."
        )


# ---------------------------------------------------------------------------
# 2. Statistical properties
# ---------------------------------------------------------------------------

class TestStatisticalProperties:
    """
    Simulated returns should recover the mean and correlation structure
    of the input data (within tolerance given finite samples).
    """

    def test_cholesky_normal_zero_mean(self, sample_returns):
        """CholeskyNormal simulates zero-mean returns (standard practice)."""
        sim = CholeskyNormal()
        result = sim.simulate(sample_returns, N_SIMS)
        col_means = np.abs(result.mean(axis=0))
        assert np.all(col_means < 0.01), (
            f"CholeskyNormal means deviate too much from zero: {col_means}"
        )

    def test_cholesky_normal_preserves_vol(self, sample_returns):
        """
        Simulated vol should be close to historical vol.
        Tolerance: ±30% relative (finite sample noise).
        """
        sim = CholeskyNormal()
        result = sim.simulate(sample_returns, N_SIMS)

        hist_vol = sample_returns.std().values
        sim_vol = result.std(axis=0)

        ratio = sim_vol / hist_vol
        assert np.all(ratio > 0.7) and np.all(ratio < 1.3), (
            f"Simulated vol deviates >30% from historical. Ratios: {ratio}"
        )

    def test_empirical_returns_are_from_history(self, sample_returns):
        """
        Empirical bootstrap must only return values that appear in
        the historical dataset (within floating-point tolerance).
        """
        sim = CholeskyEmpirical()
        result = sim.simulate(sample_returns, N_SIMS)

        hist_values = set(np.round(sample_returns.values.flatten(), 12))
        sim_values = set(np.round(result.flatten(), 12))

        # All simulated values should be drawable from history
        # (we check a sample to avoid O(n²) cost)
        sample_size = min(200, len(sim_values))
        rng = np.random.default_rng(0)
        sampled = rng.choice(list(sim_values), size=sample_size, replace=False)

        not_in_history = [v for v in sampled if v not in hist_values]
        assert len(not_in_history) == 0, (
            f"CholeskyEmpirical generated {len(not_in_history)} values "
            f"not found in historical data."
        )

    def test_pca_normal_positive_variance(self, sample_returns):
        """PCA-based simulation must produce variance > 0 for each asset."""
        sim = PCANormal()
        result = sim.simulate(sample_returns, N_SIMS)
        sim_var = result.var(axis=0)
        assert np.all(sim_var > 0), (
            f"PCANormal produced zero variance for some assets: {sim_var}"
        )

    def test_all_methods_produce_negative_scenarios(self, sample_returns):
        """Risk simulations must include loss scenarios (negative returns)."""
        for SimClass in [CholeskyNormal, CholeskyEmpirical, PCANormal, PCAEmpirical]:
            sim = SimClass()
            result = sim.simulate(sample_returns, N_SIMS)
            has_losses = np.any(result < 0)
            assert has_losses, (
                f"{SimClass.__name__} produced no negative returns — "
                f"something is wrong with the simulation."
            )


# ---------------------------------------------------------------------------
# 3. Determinism with fixed seed (if simulators accept a seed)
# ---------------------------------------------------------------------------

class TestReproducibility:
    """
    Simulators that accept a random seed must produce identical results
    when called twice with the same seed.
    """

    @pytest.mark.parametrize("SimClass", [CholeskyNormal, PCANormal])
    def test_same_seed_same_result(self, SimClass, sample_returns):
        """Two runs with seed=0 should match exactly."""
        try:
            sim1 = SimClass(random_state=0)
            sim2 = SimClass(random_state=0)
        except TypeError:
            pytest.skip(f"{SimClass.__name__} does not accept random_state — skip.")

        r1 = sim1.simulate(sample_returns, 100)
        r2 = sim2.simulate(sample_returns, 100)
        np.testing.assert_array_equal(r1, r2,
            err_msg=f"{SimClass.__name__}: different results with same seed.")

    @pytest.mark.parametrize("SimClass", [CholeskyNormal, PCANormal])
    def test_different_seeds_differ(self, SimClass, sample_returns):
        """Two runs with different seeds should not be identical."""
        try:
            sim1 = SimClass(random_state=0)
            sim2 = SimClass(random_state=1)
        except TypeError:
            pytest.skip(f"{SimClass.__name__} does not accept random_state — skip.")

        r1 = sim1.simulate(sample_returns, 100)
        r2 = sim2.simulate(sample_returns, 100)
        assert not np.array_equal(r1, r2), (
            f"{SimClass.__name__}: seeds 0 and 1 produced identical results."
        )


# ---------------------------------------------------------------------------
# 4. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_single_simulation(self, sample_returns):
        """Edge: N_SIMS=1 should not crash any simulator."""
        for SimClass in [CholeskyNormal, CholeskyEmpirical, PCANormal, PCAEmpirical]:
            sim = SimClass()
            result = sim.simulate(sample_returns, 1)
            assert result.shape[0] == 1

    def test_single_asset(self):
        """Edge: 1 asset portfolio (no correlation matrix needed)."""
        rng = np.random.default_rng(0)
        single_asset = pd.DataFrame(
            rng.normal(0, 0.01, (252, 1)), columns=["AAPL"]
        )
        for SimClass in [CholeskyNormal, CholeskyEmpirical]:
            sim = SimClass()
            result = sim.simulate(single_asset, 1000)
            assert result.shape == (1000, 1)

    def test_large_simulation_does_not_explode(self, sample_returns):
        """Sanity: 50k simulations should run without memory errors."""
        sim = CholeskyNormal()
        result = sim.simulate(sample_returns, 50_000)
        assert result.shape[0] == 50_000
