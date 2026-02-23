from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOG_DIR = Path("logs")
RESULTS_CSV = LOG_DIR / "benchmark_results.csv"
HEAT_AI = LOG_DIR / "heatmap_ai.npy"
HEAT_BASE = LOG_DIR / "heatmap_baseline.npy"
OUTPUT = LOG_DIR / "thunder_comparison.png"


def plot_bar(ax, df: pd.DataFrame) -> None:
    summary = df.groupby("Agent", as_index=False)["AvgPower_kW"].mean()
    if set(["GreenHPC-RL", "Baseline"]).issubset(summary["Agent"].unique()):
        ai_power = float(summary.loc[summary["Agent"] == "GreenHPC-RL", "AvgPower_kW"].iloc[0])
        base_power = float(summary.loc[summary["Agent"] == "Baseline", "AvgPower_kW"].iloc[0])
        savings_pct = 100.0 * (base_power - ai_power) / base_power if base_power > 0 else 0.0
    else:
        ai_power = base_power = savings_pct = 0.0

    ax.bar(["Baseline", "AI"], [base_power, ai_power], color=["gray", "green"])
    ax.set_ylabel("Avg Power (kW)")
    ax.set_title("Average Power Comparison")
    ax.text(0.5, max(base_power, ai_power) * 0.95 if max(base_power, ai_power) else 0.1,
            f"Savings: {savings_pct:.1f}%", ha="center", va="top", fontsize=10)


def plot_heatmaps(ax_base, ax_ai, heat_base: np.ndarray, heat_ai: np.ndarray) -> None:
    t_base, racks_base = heat_base.shape if heat_base is not None else (0, 0)
    t_ai, racks_ai = heat_ai.shape if heat_ai is not None else (0, 0)

    im1 = ax_base.imshow(heat_base.T, aspect="auto", origin="lower", cmap="coolwarm")
    ax_base.set_title("Baseline Rack Temps")
    ax_base.set_ylabel("Rack ID")
    ax_base.set_xlabel("Time")

    im2 = ax_ai.imshow(heat_ai.T, aspect="auto", origin="lower", cmap="coolwarm")
    ax_ai.set_title("AI Rack Temps")
    ax_ai.set_ylabel("Rack ID")
    ax_ai.set_xlabel("Time")

    cbar = plt.colorbar(im2, ax=[ax_base, ax_ai], shrink=0.8, pad=0.02)
    cbar.set_label("Avg Rack Temp (C)")


def main() -> None:
    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"Missing results file: {RESULTS_CSV}")
    df = pd.read_csv(RESULTS_CSV)

    if not HEAT_AI.exists() or not HEAT_BASE.exists():
        raise FileNotFoundError("Missing heatmap npy files; run evaluate first.")

    heat_ai = np.load(HEAT_AI)
    heat_base = np.load(HEAT_BASE)

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    plot_bar(axes[0], df)

    fig_hm, hm_axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    plot_heatmaps(hm_axes[0], hm_axes[1], heat_base, heat_ai)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig_hm.tight_layout()

    fig.savefig(OUTPUT, dpi=200)
    fig_hm.savefig(OUTPUT.with_name("thunder_heatmaps.png"), dpi=200)
    print(f"Saved plots to {OUTPUT} and {OUTPUT.with_name('thunder_heatmaps.png')}")


if __name__ == "__main__":
    main()
