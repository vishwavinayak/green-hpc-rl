from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from typing import Callable, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.baseline import BaselineAgent
from src.agents.pdqn_agent import PDQNAgent
from src.envs.hybrid_dc import LLNLThunderEnv

MAX_STEPS = 200
N_EPISODES = 50
LOG_DIR = Path("logs")
QNET_PATH = LOG_DIR / "q_network.pth"


def _reset_workload(env: LLNLThunderEnv) -> None:
    """Reset workload deque so each run sees identical job sequence."""
    if hasattr(env, "workload") and hasattr(env.workload, "_tasks_template"):
        env.workload._queue = deque(env.workload._tasks_template)


def _load_pdqn(env: LLNLThunderEnv, device: torch.device) -> PDQNAgent:
    state_size = int(env.observation_space.shape[0])
    n_servers = int(env.TOTAL_SERVERS)
    action_param_size = int(env.action_space[1].shape[0])

    agent = PDQNAgent(
        state_size,
        n_servers,
        action_param_size,
        device=device,
    )

    if QNET_PATH.exists():
        state_dict = torch.load(QNET_PATH, map_location=device)
        agent.q_network.load_state_dict(state_dict)
        agent.target_q_network.load_state_dict(state_dict)
        agent.q_network.eval()
        agent.target_q_network.eval()
    else:
        print(f"Warning: {QNET_PATH} not found. Running PDQN with random weights.")

    return agent


def _run_episode(
    env: LLNLThunderEnv,
    select_action_fn: Callable[[np.ndarray], Tuple[int, np.ndarray]],
    record_heatmap: bool,
) -> tuple[float, float, int, list[np.ndarray]]:
    _reset_workload(env)
    state, _ = env.reset()

    total_reward = 0.0
    power_readings: list[float] = []
    max_temp = -np.inf
    sla_violations = 0
    heatmap: list[np.ndarray] = []

    for _ in range(MAX_STEPS):
        action_idx, action_params = select_action_fn(state)
        action = (action_idx, np.array(action_params, dtype=np.float32))

        next_state, reward, terminated, truncated, info = env.step(action)

        temps = env.temps
        step_max_temp = float(np.max(temps))
        max_temp = max(max_temp, step_max_temp)
        if step_max_temp > 30.0:
            sla_violations += 1

        power = float(info.get("p_it_sum", 0.0)) + float(info.get("p_cooling", 0.0))
        power_readings.append(power)

        if record_heatmap:
            heatmap.append(temps.copy())

        total_reward += float(reward)
        state = next_state

        if terminated or truncated:
            break

    return total_reward, max_temp, sla_violations, heatmap, power_readings


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    env_ai = LLNLThunderEnv()
    env_base = LLNLThunderEnv()

    pdqn_agent = _load_pdqn(env_ai, device=device)
    baseline_agent = BaselineAgent(action_dim=env_base.TOTAL_SERVERS)

    def pdqn_policy(obs: np.ndarray) -> Tuple[int, np.ndarray]:
        with torch.no_grad():
            idx, params = pdqn_agent.select_action(obs, epsilon=0.0)
            return idx, params.detach().cpu().numpy()

    def baseline_policy(obs: np.ndarray) -> Tuple[int, np.ndarray]:
        server_idx, airflow = baseline_agent.select_action(obs)
        return server_idx, np.array(airflow, dtype=np.float32)

    stats: list[dict] = []

    heatmap_ai: list[np.ndarray] = []
    heatmap_base: list[np.ndarray] = []

    for episode in range(1, N_EPISODES + 1):
        total_reward_ai, max_temp_ai, sla_ai, heat_ai, power_ai = _run_episode(
            env_ai, pdqn_policy, record_heatmap=episode == N_EPISODES
        )
        if heat_ai:
            heatmap_ai = heat_ai

        total_reward_base, max_temp_base, sla_base, heat_base, power_base = _run_episode(
            env_base, baseline_policy, record_heatmap=episode == N_EPISODES
        )
        if heat_base:
            heatmap_base = heat_base

        stats.append(
            {
                "Agent": "GreenHPC-RL",
                "Episode": episode,
                "TotalReward": total_reward_ai,
                "AvgPower_kW": float(np.mean(power_ai)) / 1000.0 if power_ai else 0.0,
                "MaxTemp": max_temp_ai,
                "SLA_Violations": sla_ai,
            }
        )
        stats.append(
            {
                "Agent": "Baseline",
                "Episode": episode,
                "TotalReward": total_reward_base,
                "AvgPower_kW": float(np.mean(power_base)) / 1000.0 if power_base else 0.0,
                "MaxTemp": max_temp_base,
                "SLA_Violations": sla_base,
            }
        )

    df = pd.DataFrame(stats)
    df.to_csv(LOG_DIR / "benchmark_results.csv", index=False)
    print(f"Saved benchmark results to {LOG_DIR / 'benchmark_results.csv'}")

    if heatmap_ai:
        heat_ai_arr = np.stack(heatmap_ai, axis=0)  # (T, 840)
        heat_ai_rack = heat_ai_arr.reshape(heat_ai_arr.shape[0], 20, 42).mean(axis=2)
        np.save(LOG_DIR / "heatmap_ai.npy", heat_ai_rack)
        print(f"Saved AI heatmap to {LOG_DIR / 'heatmap_ai.npy'}")

    if heatmap_base:
        heat_base_arr = np.stack(heatmap_base, axis=0)  # (T, 840)
        heat_base_rack = heat_base_arr.reshape(heat_base_arr.shape[0], 20, 42).mean(axis=2)
        np.save(LOG_DIR / "heatmap_baseline.npy", heat_base_rack)
        print(f"Saved baseline heatmap to {LOG_DIR / 'heatmap_baseline.npy'}")


if __name__ == "__main__":
    main()
