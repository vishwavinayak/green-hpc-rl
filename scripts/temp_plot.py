from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

LOG_DIR = Path("logs")
RESULTS_CSV = LOG_DIR / "benchmark_results.csv"
LOAD_AI = LOG_DIR / "load_map_ai.npy"
LOAD_BASE = LOG_DIR / "load_map_baseline.npy"
OUTPUT = LOG_DIR / "advanced_analysis.png"


def main() -> None:
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Missing results file: {RESULTS_CSV}")
    if not LOAD_AI.exists() or not LOAD_BASE.exists():
        raise FileNotFoundError(
            "Missing load map npy files; run evaluation to generate them."
        )

    df = pd.read_csv(RESULTS_CSV)
    load_ai = np.load(LOAD_AI)
    load_base = np.load(LOAD_BASE)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Subplot 1: Energy ROI bar chart.
    summary = df.groupby("Agent", as_index=False)["AvgPower_kW"].mean()
    ai_power = (
        float(summary.loc[summary["Agent"] == "GreenHPC-RL", "AvgPower_kW"].iloc[0])
        if "GreenHPC-RL" in summary["Agent"].values
        else 0.0
    )
    base_power = (
        float(summary.loc[summary["Agent"] == "Baseline", "AvgPower_kW"].iloc[0])
        if "Baseline" in summary["Agent"].values
        else 0.0
    )
    savings_pct = (
        100.0 * (base_power - ai_power) / base_power if base_power > 0 else 0.0
    )
    axes[0, 0].bar(["Baseline", "AI"], [base_power, ai_power], color=["gray", "green"])
    axes[0, 0].set_ylabel("Avg Power (kW)")
    axes[0, 0].set_title("Energy ROI")
    axes[0, 0].text(
        0.5,
        max(base_power, ai_power) * 0.9 if max(base_power, ai_power) else 0.1,
        f"Savings: {savings_pct:.1f}%",
        ha="center",
        va="top",
        fontsize=12,
        color="black",
    )

    # Subplot 2: Workload verification (total load over time).
    total_load_base = load_base.sum(axis=1)
    total_load_ai = load_ai.sum(axis=1)
    diff = np.sum(np.abs(total_load_base - total_load_ai))
    print(f"Total load absolute difference (Baseline vs AI): {diff}")

    axes[0, 1].plot(
        total_load_base,
        label="Baseline Total Load",
        color="gray",
        linewidth=4,
    )
    axes[0, 1].plot(
        total_load_ai,
        label="AI Total Load",
        color="green",
        linewidth=2,
        linestyle="--",
    )
    axes[0, 1].set_title("Workload Verification")
    axes[0, 1].set_xlabel("Time")
    axes[0, 1].set_ylabel("Total Load")
    axes[0, 1].legend()

    # Subplot 3 and 4: Heatmaps of load distribution per rack.
    vmax = 0.6
    im0 = axes[1, 0].imshow(
        load_base.T, aspect="auto", origin="lower", cmap="inferno", vmax=vmax
    )
    axes[1, 0].set_title("Baseline Distribution (Concentrated)")
    axes[1, 0].set_ylabel("Rack ID")
    axes[1, 0].set_xlabel("Time")

    im1 = axes[1, 1].imshow(
        load_ai.T, aspect="auto", origin="lower", cmap="inferno", vmax=vmax
    )
    axes[1, 1].set_title("GreenHPC Distribution (Sparse)")
    axes[1, 1].set_ylabel("Rack ID")
    axes[1, 1].set_xlabel("Time")

    cbar = fig.colorbar(im1, ax=[axes[1, 0], axes[1, 1]], shrink=0.8, pad=0.02)
    cbar.set_label("Avg Rack Load")

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=200)
    print(f"Saved figure to {OUTPUT}")


if __name__ == "__main__":
    main()
