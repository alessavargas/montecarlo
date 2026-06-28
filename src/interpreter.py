"""
src/interpreter.py

Financial interpretation layer for Monte Carlo VaR/CVaR results.

Adds context to raw numbers: what the estimates mean, how to read
method divergence, and what assumptions drive each result.
"""

from typing import Dict
import numpy as np


class FinancialInterpreter:
    """
    Interprets VaR/CVaR results from a financial risk perspective.

    Parameters
    ----------
    results : dict
        Output from main.py — keys are method names, values are dicts
        with "VaR", "CVaR", and "PnL" keys.
    config : dict
        The CONFIG dict from src/config.py.
    """

    def __init__(self, results: Dict, config: Dict):
        self.results = results
        self.config = config
        self.confidence = config["confidence_level"]
        self.horizon = config.get("horizon_days", 1)
        self.method_names = list(results.keys())

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def full_report(self) -> None:
        """Print the complete financial interpretation."""
        self._section("FINANCIAL INTERPRETATION")
        self._interpret_var()
        self._interpret_cvar()
        self._interpret_tail_ratio()
        self._compare_methods()
        self._model_risk()
        self._print_limitations()

    # ------------------------------------------------------------------
    # Individual sections
    # ------------------------------------------------------------------

    def _interpret_var(self) -> None:
        """Explain VaR in plain financial language."""
        print("\n── Value at Risk (VaR) ──────────────────────────────────────")
        pct = int(self.confidence * 100)
        alpha = int((1 - self.confidence) * 100)

        for name, res in self.results.items():
            var = res["VaR"]
            print(
                f"  {name:<30} VaR({pct}%) = ${var:,.0f}\n"
                f"  {'':30} → On a {self.horizon}-day horizon, losses exceed "
                f"${abs(var):,.0f} only {alpha}% of the time."
            )

    def _interpret_cvar(self) -> None:
        """Explain CVaR and its relationship to VaR."""
        print("\n── Conditional VaR / Expected Shortfall ────────────────────")
        pct = int(self.confidence * 100)

        for name, res in self.results.items():
            cvar = res["CVaR"]
            var = res["VaR"]
            excess = abs(cvar) - abs(var)
            print(
                f"  {name:<30} CVaR({pct}%) = ${cvar:,.0f}\n"
                f"  {'':30} → In the worst {100-pct}% of scenarios, "
                f"average loss is ${abs(cvar):,.0f}.\n"
                f"  {'':30} → That is ${excess:,.0f} worse than VaR alone suggests."
            )

    def _interpret_tail_ratio(self) -> None:
        """
        CVaR/VaR ratio: how thick is the tail?
        Ratio near 1.0 → thin tail (Normal-like).
        Ratio > 1.5   → heavy tail (fat tails present).
        """
        print("\n── Tail Thickness (CVaR / VaR ratio) ───────────────────────")
        print(
            "  Ratio ≈ 1.0–1.2 → thin tail, close to Normal\n"
            "  Ratio > 1.5     → heavy tail, risk is concentrated beyond VaR\n"
        )
        for name, res in self.results.items():
            ratio = abs(res["CVaR"]) / abs(res["VaR"]) if res["VaR"] != 0 else float("nan")
            tag = "⚠ heavy tail" if ratio > 1.5 else "✓ moderate tail"
            print(f"  {name:<30} {ratio:.3f}  ({tag})")

    def _compare_methods(self) -> None:
        """Flag when methods diverge significantly — that divergence is model risk."""
        print("\n── Method Comparison ───────────────────────────────────────")
        var_values = {k: abs(v["VaR"]) for k, v in self.results.items()}
        min_var = min(var_values.values())
        max_var = max(var_values.values())
        spread_pct = (max_var - min_var) / min_var * 100 if min_var != 0 else 0

        print(f"  VaR range across methods: ${min_var:,.0f} – ${max_var:,.0f}")
        print(f"  Spread: {spread_pct:.1f}% of minimum estimate")

        if spread_pct > 20:
            print(
                "\n  ⚠  High model risk detected (>20% spread).\n"
                "     The choice of simulation method materially affects your\n"
                "     risk estimate. Consider using the most conservative method\n"
                "     (highest VaR) for capital allocation decisions."
            )
        elif spread_pct > 10:
            print(
                "\n  ○  Moderate model risk (10–20% spread).\n"
                "     Methods broadly agree. Empirical methods likely capture\n"
                "     fat tails that the Normal variants miss."
            )
        else:
            print(
                "\n  ✓  Low model risk (<10% spread).\n"
                "     All methods converge — the distributional assumption\n"
                "     is not driving your risk estimate."
            )

    def _model_risk(self) -> None:
        """Recommend which method to trust based on the data."""
        print("\n── Method Recommendation ───────────────────────────────────")
        pnl_series = list(self.results.values())[0]["PnL"]

        # Simple kurtosis check to flag fat tails
        if len(pnl_series) > 10:
            excess_kurtosis = float(np.mean((pnl_series - np.mean(pnl_series)) ** 4)
                                    / np.std(pnl_series) ** 4) - 3
        else:
            excess_kurtosis = 0

        print(f"  Excess kurtosis of PnL (Cholesky-Normal): {excess_kurtosis:.2f}")

        if excess_kurtosis > 1.0:
            print(
                "  → Fat tails detected. Prefer Empirical methods (bootstrap).\n"
                "     Normal-based methods underestimate tail risk here."
            )
        else:
            print(
                "  → PnL distribution is near-Normal.\n"
                "     All methods should give comparable results."
            )

    def _print_limitations(self) -> None:
        """Print model limitations — important for interviews and presentations."""
        print("\n── Model Limitations ───────────────────────────────────────")
        limitations = [
            ("Linear correlation only",
             "All methods use covariance/PCA. Tail dependence (co-crashes) is not modeled."),
            ("Historical data dependency",
             "Empirical bootstrap assumes history is representative. It misses unseen tail events."),
            ("Normality (Normal variants)",
             "Cholesky/PCA-Normal underestimate VaR when returns are fat-tailed."),
            ("Horizon scaling",
             f"{self.horizon}-day VaR uses sqrt(T) scaling, which assumes i.i.d. daily returns."),
            ("No liquidity adjustment",
             "VaR assumes positions close at mid-price. Stressed markets widen spreads."),
            ("Static portfolio",
             "No intraday rebalancing or delta hedging is modeled."),
        ]
        for title, detail in limitations:
            print(f"\n  [{title}]\n  {detail}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _section(title: str) -> None:
        bar = "=" * 70
        print(f"\n{bar}\n  {title}\n{bar}")
