"""Reporting and visualization"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Reporter:
    """Generate reports and visualizations"""
    
    def __init__(self, results, config):
        self.results = results
        self.config = config
    
    def print_summary(self):
        """Print summary table"""
        print("\n" + "=" * 80)
        print(f"VaR/CVaR Summary ({int(self.config['confidence_level']*100)}% confidence, {self.config['horizon_days']} day)")
        print("=" * 80)
        
        summary_data = []
        for method, data in self.results.items():
            summary_data.append({
                "Method": method,
                "VaR": f"${data['VaR']:,.2f}",
                "CVaR": f"${data['CVaR']:,.2f}",
            })
        
        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False))
    
    def export_to_csv(self, filename):
        """Export results to CSV"""
        summary_data = []
        for method, data in self.results.items():
            summary_data.append({
                "Method": method,
                "VaR": data["VaR"],
                "CVaR": data["CVaR"],
            })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(filename, index=False)
        print(f"\nResults exported to {filename}")
    
    def plot_distributions(self):
        """Plot P&L distributions"""
        n_methods = len(self.results)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for ax, (method, data) in zip(axes, self.results.items()):
            pnl = data["PnL"]
            ax.hist(pnl, bins=50, alpha=0.7, edgecolor="black", color="steelblue")
            ax.axvline(-data["VaR"], color="red", linestyle="--", linewidth=2, label=f"VaR: ${-data['VaR']:,.0f}")
            ax.axvline(-data["CVaR"], color="orange", linestyle="--", linewidth=2, label=f"CVaR: ${-data['CVaR']:,.0f}")
            ax.set_title(f"{method}", fontsize=12, fontweight="bold")
            ax.set_xlabel("P&L ($)")
            ax.set_ylabel("Frequency")
            ax.legend(loc="upper left")
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("pnl_distributions.png", dpi=300, bbox_inches="tight")
        print("Distribution plot saved to pnl_distributions.png")
        plt.close()
