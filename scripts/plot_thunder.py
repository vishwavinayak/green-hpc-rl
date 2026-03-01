from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOG_DIR = Path("logs")
RESULTS_CSV = LOG_DIR / "benchmark_results.csv"
HEAT_AI = LOG_DIR / "load_map_ai.npy"
HEAT_BASE = LOG_DIR / "load_map_baseline.npy"
OUTPUT = LOG_DIR / "thunder_comparison.png"


def plot_bar(ax, df: pd.DataFrame) -> None:
    summary = df.groupby("Agent", as_index=False)["AvgPower_kW"].mean()

    # Defaults
    ai_power = 0.0
    base_power = 0.0
    savings_pct = 0.0

    if not summary.empty:
        # Check if agents exist in the summary
        if "GreenHPC-RL" in summary["Agent"].values:
            ai_power = float(
                summary.loc[summary["Agent"] == "GreenHPC-RL", "AvgPower_kW"].iloc[0]
            )
        if "Baseline" in summary["Agent"].values:
            base_power = float(
                summary.loc[summary["Agent"] == "Baseline", "AvgPower_kW"].iloc[0]
            )

        if base_power > 0:
            savings_pct = 100.0 * (base_power - ai_power) / base_power

    ax.bar(
        ["Baseline", "GreenHPC-RL"],
        [base_power, ai_power],
        color=["#4e6b8a", "#42a875"],
    )
    ax.set_ylabel("Avg Power (kW)")
    ax.set_title("Energy Efficiency Battle")

    # Add text label
    height = max(base_power, ai_power)
    ax.text(
        0.5,
        height * 0.9 if height > 0 else 0.1,
        f"Savings: {savings_pct:.1f}%",
        ha="center",
        va="top",
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
    )


def plot_heatmaps(ax_base, ax_ai, heat_base: np.ndarray, heat_ai: np.ndarray) -> None:
    # Plot Baseline
    im1 = ax_base.imshow(
        heat_base.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    ax_base.set_title("Baseline Load Distribution")
    ax_base.set_ylabel("Rack ID")
    ax_base.set_xlabel("Time Step")

    # Plot AI
    im2 = ax_ai.imshow(
        heat_ai.T,
        aspect="auto",
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    ax_ai.set_title("GreenHPC-RL Load Distribution")
    ax_ai.set_ylabel("Rack ID")
    ax_ai.set_xlabel("Time Step")

    # Colorbar
    cbar = plt.colorbar(im2, ax=[ax_base, ax_ai], shrink=0.8, pad=0.02)
    cbar.set_label("Rack CPU Utilization (0-1)")


def main() -> None:
    if not RESULTS_CSV.exists():
        print(f"File not found: {RESULTS_CSV}")
        return
    df = pd.read_csv(RESULTS_CSV)

    if not HEAT_AI.exists() or not HEAT_BASE.exists():
        print("Missing heatmap npy files")
        return

    heat_ai = np.load(HEAT_AI)
    heat_base = np.load(HEAT_BASE)

    # Create a layout with 2 rows: Top for Bar, Bottom for Heatmaps
    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.5])

    # Bar Plot (Spans both columns at top)
    ax_bar = fig.add_subplot(gs[0, :])
    plot_bar(ax_bar, df)

    # Heatmaps (Side by side at bottom)
    ax_base = fig.add_subplot(gs[1, 0])
    ax_ai = fig.add_subplot(gs[1, 1], sharey=ax_base)

    plot_heatmaps(ax_base, ax_ai, heat_base, heat_ai)

    fig.savefig(OUTPUT, dpi=150)
    print(f"✅ Saved comparison plot to {OUTPUT}")


if __name__ == "__main__":
    main()
