from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import gymnasium as gym
import numpy as np
import torch  # noqa: F401 - imported for P-DQN compatibility

from .physics import MAX_POWER, ThermalPhysics
from .workload import WorkloadGenerator


class HybridDataCenterEnv(gym.Env):
    """Hybrid data center environment for P-DQN control."""

    metadata = {"render_modes": []}

    def __init__(
        self, workload_path: str | Path = "data/raw/borg_traces_data.csv"
    ) -> None:
        super().__init__()
        self.physics = ThermalPhysics()
        self.workload = WorkloadGenerator(str(workload_path))
        self.n_servers = 10

        obs_dim = self.n_servers * 2 + 1
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Tuple(
            (
                gym.spaces.Discrete(self.n_servers),
                gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            )
        )

        self.server_loads = np.zeros(self.n_servers, dtype=np.float32)
        self.server_temps = np.full(self.n_servers, 20.0, dtype=np.float32)
        self.next_job_size = float(self.workload.step())

    def _get_obs(self) -> np.ndarray:
        temps_norm = np.clip(self.server_temps / 100.0, 0.0, 1.0)
        return np.concatenate(
            [
                self.server_loads,
                temps_norm,
                np.array([self.next_job_size], dtype=np.float32),
            ]
        ).astype(np.float32)

    def step(self, action: Tuple[int, Any]):
        server_idx, airflow = action
        airflow_value = float(np.array(airflow).reshape(-1)[0])
        airflow_value = float(np.clip(airflow_value, 0.0, 1.0))

        # Assign current job to the chosen server.
        job_size = float(self.next_job_size)
        self.server_loads[server_idx] = float(
            np.clip(self.server_loads[server_idx] + job_size, 0.0, 1.0)
        )

        it_power = 0.0
        for i in range(self.n_servers):
            server_power = self.physics.calculate_power(float(self.server_loads[i]))
            self.server_temps[i] = float(
                self.physics.update_temperature(
                    float(self.server_temps[i]), server_power, airflow_value
                )
            )
            it_power += server_power

        # Simple cooling cost proportional to airflow and rack max power budget.
        cooling_power = airflow_value * MAX_POWER
        total_power = it_power + cooling_power

        penalty = 1.0 if np.any(self.server_temps > 30.0) else 0.0
        reward = -total_power - 100.0 * penalty

        # Fetch next job for the following decision.
        self.next_job_size = float(self.workload.step())

        obs = self._get_obs()
        terminated = False
        truncated = False
        info = {
            "it_power": it_power,
            "cooling_power": cooling_power,
            "penalty": penalty,
            "total_power": total_power,
        }
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.server_loads = np.zeros(self.n_servers, dtype=np.float32)
        self.server_temps = np.full(self.n_servers, 20.0, dtype=np.float32)
        self.next_job_size = float(self.workload.step())
        return self._get_obs(), {}

    def render(self):  # pragma: no cover - not required
        return None
