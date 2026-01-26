import sys
from pathlib import Path
from typing import Callable, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.baseline import BaselineAgent
from src.agents.pdqn_agent import PDQNAgent
from src.envs.hybrid_dc import HybridDataCenterEnv

MAX_STEPS = 200
N_EPISODES = 50
LOG_DIR = Path("logs")
QNET_PATH = LOG_DIR / "q_network.pth"


def _reset_workload(env: HybridDataCenterEnv, start_index: int) -> None:
    """Align workload generator to a given starting index."""
    if hasattr(env, "workload") and hasattr(env.workload, "_index"):
        data_len = len(env.workload)
        env.workload._index = start_index % max(data_len, 1)


def _load_pdqn(env: HybridDataCenterEnv) -> PDQNAgent:
    state_size = env.observation_space.shape[0]
    n_servers = env.n_servers
    action_param_size = env.action_space[1].shape[0]

    agent = PDQNAgent(state_size, n_servers, action_param_size)
    if QNET_PATH.exists():
        state_dict = torch.load(QNET_PATH, map_location="cpu")
        agent.q_network.load_state_dict(state_dict)
        agent.target_q_network.load_state_dict(state_dict)
        agent.q_network.eval()
        agent.target_q_network.eval()
    else:
        print(f"Warning: {QNET_PATH} not found. Running PDQN with random weights.")
    return agent


def _run_episode(
    env: HybridDataCenterEnv,
    select_action_fn: Callable[[np.ndarray], Tuple[int, Sequence[float]]],
    record_heatmap: bool,
    start_index: int,
    n_servers: int,
) -> tuple[float, float, int, list[np.ndarray]]:
    _reset_workload(env, start_index)
    state, _ = env.reset()

    total_energy = 0.0
    max_temp = -np.inf
    sla_violations = 0
    heatmap: list[np.ndarray] = []

    for _ in range(MAX_STEPS):
        action_idx, action_params = select_action_fn(state)
        action = (action_idx, np.array(action_params, dtype=np.float32))

        next_state, _, _, _, info = env.step(action)

        temps = env.server_temps.copy()
        max_temp = max(max_temp, float(np.max(temps)))
        if np.any(temps > 30.0):
            sla_violations += 1

        total_energy += float(info.get("total_power", 0.0))

        if record_heatmap:
            heatmap.append(temps)

        state = next_state

    return total_energy, max_temp, sla_violations, heatmap


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    env_ai = HybridDataCenterEnv()
    env_base = HybridDataCenterEnv()

    pdqn_agent = _load_pdqn(env_ai)
    baseline_agent = BaselineAgent(env_ai.n_servers)

    def pdqn_policy(obs: np.ndarray) -> Tuple[int, Sequence[float]]:
        with torch.no_grad():
            idx, params = pdqn_agent.select_action(obs, epsilon=0.0)
            return idx, params.cpu().numpy()

    def baseline_policy(obs: np.ndarray) -> Tuple[int, Sequence[float]]:
        adjusted = obs.copy()
        adjusted[10 : 10 + baseline_agent.n_servers] = env_base.server_temps
        return baseline_agent.select_action(adjusted)

    stats: list[dict] = []

    heatmap_ai: list[np.ndarray] = []
    heatmap_base: list[np.ndarray] = []

    for episode in range(1, N_EPISODES + 1):
        # Use identical workload start for both agents each episode.
        workload_start = 0

        total_energy_ai, max_temp_ai, sla_ai, heat_ai = _run_episode(
            env_ai,
            pdqn_policy,
            record_heatmap=episode == N_EPISODES,
            start_index=workload_start,
            n_servers=env_ai.n_servers,
        )
        stats.append(
            {
                "Agent_Type": "GreenHPC-RL",
                "Episode": episode,
                "Total_Energy": total_energy_ai,
                "Max_Temp": max_temp_ai,
                "SLA_Violations": sla_ai,
            }
        )
        if heat_ai:
            heatmap_ai = heat_ai

        total_energy_base, max_temp_base, sla_base, heat_base = _run_episode(
            env_base,
            baseline_policy,
            record_heatmap=episode == N_EPISODES,
            start_index=workload_start,
            n_servers=env_base.n_servers,
        )
        stats.append(
            {
                "Agent_Type": "Baseline",
                "Episode": episode,
                "Total_Energy": total_energy_base,
                "Max_Temp": max_temp_base,
                "SLA_Violations": sla_base,
            }
        )
        if heat_base:
            heatmap_base = heat_base

    df = pd.DataFrame(stats)
    df.to_csv(LOG_DIR / "benchmark_stats.csv", index=False)
    print(f"Saved benchmark stats to {LOG_DIR / 'benchmark_stats.csv'}")

    if heatmap_ai:
        np.save(LOG_DIR / "heatmap_data_ai.npy", np.stack(heatmap_ai, axis=0))
        print(f"Saved AI heatmap to {LOG_DIR / 'heatmap_data_ai.npy'}")
    if heatmap_base:
        np.save(LOG_DIR / "heatmap_data_baseline.npy", np.stack(heatmap_base, axis=0))
        print(f"Saved baseline heatmap to {LOG_DIR / 'heatmap_data_baseline.npy'}")


if __name__ == "__main__":
    main()
