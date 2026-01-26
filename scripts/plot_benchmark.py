from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

LOG_DIR = Path("logs")
STATS_PATH = LOG_DIR / "benchmark_stats.csv"
HEAT_AI_PATH = LOG_DIR / "heatmap_data_ai.npy"
HEAT_BASE_PATH = LOG_DIR / "heatmap_data_baseline.npy"
OUTPUT_PATH = LOG_DIR / "final_comparison.png"


def plot_energy_bars(df: pd.DataFrame, ax: plt.Axes) -> None:
    grouped = df.groupby("Agent_Type")["Total_Energy"].mean().reset_index()
    grouped["Total_Energy"] = grouped["Total_Energy"] / 1000.0
    sns.barplot(
        data=grouped,
        x="Agent_Type",
        y="Total_Energy",
        hue="Agent_Type",
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.set_title("Energy Efficiency Comparison")
    ax.set_ylabel("Cumulative Power (kW)")
    ax.set_xlabel("")

    if len(grouped) == 2:
        ai_row = grouped[grouped["Agent_Type"] != "Baseline"].iloc[0]
        base_row = grouped[grouped["Agent_Type"] == "Baseline"].iloc[0]
        savings = (
            (base_row["Total_Energy"] - ai_row["Total_Energy"])
            / base_row["Total_Energy"]
            * 100
        )
        ax.text(
            0.5,
            ax.get_ylim()[1] * 0.9,
            f"Energy Savings: {savings:.2f}%",
            ha="center",
            va="center",
            fontsize=11,
            bbox=dict(boxstyle="round", fc="white", ec="black"),
        )

    for i, val in enumerate(grouped["Total_Energy"]):
        ax.text(i, val, f"{val:.1f}", ha="center", va="bottom", fontsize=9)


def plot_heatmaps(
    heat_base: np.ndarray,
    heat_ai: np.ndarray,
    ax_base: plt.Axes,
    ax_ai: plt.Axes,
    cax: plt.Axes,
) -> None:
    im0 = ax_base.imshow(heat_base.T, aspect="auto", cmap="coolwarm")
    ax_base.set_title("Baseline Thermal Map")
    ax_base.set_ylabel("Server ID")
    ax_base.set_xlabel("Time Step")

    im1 = ax_ai.imshow(heat_ai.T, aspect="auto", cmap="coolwarm")
    ax_ai.set_title("GreenHPC-RL Thermal Map")
    ax_ai.set_ylabel("Server ID")
    ax_ai.set_xlabel("Time Step")

    cbar = plt.colorbar(im1, cax=cax)
    cbar.set_label("Relative Thermal Intensity (0-1)")


def main() -> None:
    if not STATS_PATH.exists():
        raise FileNotFoundError(f"Missing stats file: {STATS_PATH}")

    df = pd.read_csv(STATS_PATH)
    if df.empty:
        raise ValueError("benchmark_stats.csv is empty")

    if not HEAT_AI_PATH.exists() or not HEAT_BASE_PATH.exists():
        raise FileNotFoundError(
            "Heatmap numpy files are missing. Run evaluate.py first."
        )

    heat_ai = np.load(HEAT_AI_PATH)
    heat_base = np.load(HEAT_BASE_PATH)

    global_max = max(float(heat_ai.max()), float(heat_base.max()))
    global_min = min(float(heat_ai.min()), float(heat_base.min()))
    denom = global_max - global_min if global_max != global_min else 1.0
    heat_ai = (heat_ai - global_min) / denom
    heat_base = (heat_base - global_min) / denom

    sns.set_theme(style="whitegrid")

    fig = plt.figure(figsize=(12, 10), layout="constrained")
    gs = fig.add_gridspec(
        2, 3, height_ratios=[1, 2], width_ratios=[1, 1, 0.05], hspace=0.3, wspace=0.2
    )

    ax_bar = fig.add_subplot(gs[0, :2])
    plot_energy_bars(df, ax_bar)

    ax_base = fig.add_subplot(gs[1, 0])
    ax_ai = fig.add_subplot(gs[1, 1])
    cax = fig.add_subplot(gs[1, 2])
    plot_heatmaps(heat_base, heat_ai, ax_base, ax_ai, cax)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=200)
    plt.close("all")
    print(f"Saved comparison figure to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
