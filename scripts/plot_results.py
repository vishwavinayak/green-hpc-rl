from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    csv_path = Path("logs/training_results.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    df = pd.read_csv(csv_path)
    if "episode" not in df.columns or "reward" not in df.columns:
        raise ValueError("CSV must contain 'episode' and 'reward' columns")

    df = df.sort_values("episode")
    df["reward_ma"] = df["reward"].rolling(window=50, min_periods=1).mean()

    plt.figure(figsize=(10, 6))
    plt.plot(df["episode"], df["reward"], color="#66b3ff", alpha=0.3, label="Reward")
    plt.plot(
        df["episode"],
        df["reward_ma"],
        color="#1f4e79",
        linewidth=2.5,
        label="50-Ep Moving Avg",
    )
    plt.title("GreenHPC-RL: Training Convergence")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True, alpha=0.2)

    output_path = Path("logs/learning_curve.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    main()
