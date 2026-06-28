"""
src/charts.py

Publication-quality charts for the Monte Carlo VaR/CVaR engine.
Designed to generate visuals that go into the README and reports.

Usage (add to main.py after running simulations):

    from src.charts import RiskCharts
    charts = RiskCharts(results, CONFIG)
    charts.save_all("docs/")          # saves PNG files for the README
    charts.show_all()                 # interactive display
"""

from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os


# Color palette — consistent across all charts
COLORS = {
    "Cholesky - Normal":    "#2563EB",   # blue
    "Cholesky - Empirical": "#16A34A",   # green
    "PCA - Normal":         "#DC2626",   # red
    "PCA - Empirical":      "#D97706",   # amber
}
ALPHA_HIST = 0.45


class RiskCharts:
    """
    Generates all visualizations for the Monte Carlo VaR/CVaR engine.

    Parameters
    ----------
    results : dict
        Output from main.py — keys are method names, values contain
        "VaR", "CVaR", and "PnL".
    config : dict
        The CONFIG dict from src/config.py.
    """

    def __init__(self, results: Dict, config: Dict):
        self.results = results
        self.config = config
        self.confidence = config["confidence_level"]
        self.method_names = list(results.keys())

        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "figure.dpi": 150,
        })

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_all(self, output_dir: str = "docs") -> None:
        """Save all charts as PNG files."""
        os.makedirs(output_dir, exist_ok=True)
        self.pnl_distributions(save_path=os.path.join(output_dir, "var_distribution.png"))
        self.method_comparison(save_path=os.path.join(output_dir, "method_comparison.png"))
        self.tail_detail(save_path=os.path.join(output_dir, "tail_detail.png"))
        print(f"Charts saved to {output_dir}/")

    def show_all(self) -> None:
        """Display all charts interactively."""
        self.pnl_distributions()
        self.method_comparison()
        self.tail_detail()
        plt.show()

    # ------------------------------------------------------------------
    # Chart 1: PnL distributions — the signature chart for the README
    # ------------------------------------------------------------------

    def pnl_distributions(self, save_path: str = None) -> plt.Figure:
        """
        Overlaid PnL histograms for all 4 methods with VaR and CVaR lines.

        This is the main README chart. Shows:
        - Distribution shape differences across methods
        - Where VaR and CVaR sit relative to the bulk of outcomes
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False, sharey=False)
        fig.suptitle(
            f"PnL Distributions — Monte Carlo VaR/CVaR\n"
            f"Portfolio: {', '.join(self.config['tickers'])}  |  "
            f"Confidence: {int(self.confidence * 100)}%  |  "
            f"Simulations: {self.config['num_simulations']:,}",
            fontsize=13, fontweight="bold", y=1.01,
        )

        for ax, (method, res) in zip(axes.flat, self.results.items()):
            pnl = res["PnL"]
            var = res["VaR"]
            cvar = res["CVaR"]
            color = COLORS.get(method, "#6B7280")

            # Histogram
            ax.hist(
                pnl, bins=80, color=color, alpha=ALPHA_HIST,
                edgecolor="white", linewidth=0.3,
            )

            # VaR line
            ax.axvline(
                var, color=color, linewidth=2.0, linestyle="--",
                label=f"VaR = ${var:,.0f}",
            )
            # CVaR line
            ax.axvline(
                cvar, color=color, linewidth=2.0, linestyle=":",
                label=f"CVaR = ${cvar:,.0f}",
            )

            # Shade the tail
            tail_x = pnl[pnl <= var]
            if len(tail_x) > 0:
                ax.hist(
                    tail_x, bins=40, color=color, alpha=0.8,
                    edgecolor="white", linewidth=0.3,
                )

            ax.set_title(method, fontsize=10, fontweight="bold")
            ax.set_xlabel("PnL ($)", fontsize=9)
            ax.set_ylabel("Frequency", fontsize=9)
            ax.legend(fontsize=8, framealpha=0.7)
            ax.yaxis.set_tick_params(labelsize=8)
            ax.xaxis.set_tick_params(labelsize=8)

            # Format x-axis ticks as currency
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
            )

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"Saved: {save_path}")
        return fig

    # ------------------------------------------------------------------
    # Chart 2: Method comparison — model risk visualization
    # ------------------------------------------------------------------

    def method_comparison(self, save_path: str = None) -> plt.Figure:
        """
        Side-by-side bar chart comparing VaR and CVaR across all 4 methods.

        Makes model risk visible: when bars differ significantly, the
        choice of simulation method is driving the risk estimate.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            "Model Risk: VaR & CVaR by Simulation Method",
            fontsize=13, fontweight="bold",
        )

        methods = self.method_names
        var_vals = [abs(self.results[m]["VaR"]) for m in methods]
        cvar_vals = [abs(self.results[m]["CVaR"]) for m in methods]
        colors = [COLORS.get(m, "#6B7280") for m in methods]
        x = np.arange(len(methods))

        # VaR chart
        bars1 = ax1.bar(x, var_vals, color=colors, alpha=0.85, width=0.6, edgecolor="white")
        ax1.set_title(f"Value at Risk ({int(self.confidence*100)}%)", fontsize=11)
        ax1.set_ylabel("Loss ($)", fontsize=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels(
            [m.replace(" - ", "\n") for m in methods], fontsize=9
        )
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        self._add_value_labels(ax1, bars1)

        # CVaR chart
        bars2 = ax2.bar(x, cvar_vals, color=colors, alpha=0.85, width=0.6, edgecolor="white")
        ax2.set_title(f"Conditional VaR / Expected Shortfall ({int(self.confidence*100)}%)", fontsize=11)
        ax2.set_ylabel("Loss ($)", fontsize=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels(
            [m.replace(" - ", "\n") for m in methods], fontsize=9
        )
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        self._add_value_labels(ax2, bars2)

        # Add model risk annotation
        spread_pct = (max(var_vals) - min(var_vals)) / min(var_vals) * 100
        fig.text(
            0.5, -0.04,
            f"Model risk spread (VaR): {spread_pct:.1f}% — "
            f"{'⚠ High' if spread_pct > 20 else '○ Moderate' if spread_pct > 10 else '✓ Low'}",
            ha="center", fontsize=10, color="#374151",
        )

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"Saved: {save_path}")
        return fig

    # ------------------------------------------------------------------
    # Chart 3: Tail detail — zoom into the loss tail
    # ------------------------------------------------------------------

    def tail_detail(self, save_path: str = None) -> plt.Figure:
        """
        Zoom into the left tail of each method's PnL distribution.

        Shows CVaR (the shaded tail beyond VaR) — the risk that
        standard VaR reporting hides.
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.set_title(
            "Tail Detail: Loss Scenarios Beyond VaR\n"
            f"(Worst {int((1 - self.confidence) * 100)}% of simulations — "
            "the region CVaR measures)",
            fontsize=12, fontweight="bold",
        )

        # Determine common x-range from all tails
        all_pnl = np.concatenate([r["PnL"] for r in self.results.values()])
        x_min = np.percentile(all_pnl, 0.1)
        x_max = np.percentile(all_pnl, (1 - self.confidence) * 100 + 2)

        for method, res in self.results.items():
            pnl = res["PnL"]
            var = res["VaR"]
            cvar = res["CVaR"]
            color = COLORS.get(method, "#6B7280")

            # KDE-style histogram of tail
            tail = pnl[pnl <= var]
            ax.hist(
                tail, bins=40, color=color, alpha=0.5,
                range=(x_min, x_max), label=method,
                edgecolor="white", linewidth=0.3,
            )
            ax.axvline(cvar, color=color, linewidth=1.5, linestyle="--", alpha=0.8)

        ax.set_xlabel("PnL ($) — loss region", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax.legend(fontsize=9, framealpha=0.8)

        # Annotation
        ax.text(
            0.02, 0.95,
            "Dashed lines = CVaR (avg loss in tail)",
            transform=ax.transAxes,
            fontsize=9, color="#6B7280", va="top",
        )

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"Saved: {save_path}")
        return fig

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _add_value_labels(ax: plt.Axes, bars) -> None:
        """Add dollar value labels on top of each bar."""
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"${height:,.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
            )
