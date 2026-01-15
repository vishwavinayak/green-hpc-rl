import os
import sys
from pathlib import Path

import torch
import pandas as pd

# Ensure project root is on PYTHONPATH when running as a script.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.pdqn_agent import PDQNAgent
from src.envs.hybrid_dc import HybridDataCenterEnv

MAX_EPISODES = 500
MAX_STEPS = 200
BATCH_SIZE = 32


def ensure_log_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_rewards_csv(rewards: list[float], path: Path) -> None:
    ensure_log_dir(path.parent)
    with path.open("w") as f:
        f.write("episode,total_reward\n")
        for idx, reward in enumerate(rewards, start=1):
            f.write(f"{idx},{reward}\n")


def main() -> None:
    env = HybridDataCenterEnv()
    state_size = env.observation_space.shape[0]
    n_servers = env.n_servers
    action_param_size = env.action_space[1].shape[0]

    agent = PDQNAgent(state_size, n_servers, action_param_size)

    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995

    rewards_log: list[float] = []
    results: list[dict] = []
    log_dir = Path("logs")
    ensure_log_dir(log_dir)

    for episode in range(1, MAX_EPISODES + 1):
        state, _ = env.reset()
        total_reward = 0.0

        for _ in range(MAX_STEPS):
            action_idx, action_params = agent.select_action(state, epsilon)
            action = (
                action_idx,
                action_params.detach().cpu().numpy(),
            )

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.buffer.push(state, action_idx, action[1], reward, next_state, done)
            agent.update(BATCH_SIZE)

            state = next_state
            total_reward += reward

            if done:
                break

        rewards_log.append(total_reward)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        print(
            f"Episode {episode} | Total Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}"
        )

        results.append({"episode": episode, "reward": total_reward, "epsilon": epsilon})

        if episode % 50 == 0:
            torch.save(agent.q_network.state_dict(), log_dir / "q_network.pth")

    save_rewards_csv(rewards_log, log_dir / "rewards.csv")

    df = pd.DataFrame(results)
    ensure_log_dir(log_dir)
    df.to_csv(log_dir / "training_results.csv", index=False)


if __name__ == "__main__":
    main()
