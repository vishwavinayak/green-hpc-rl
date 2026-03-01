from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import gymnasium as gym
import numpy as np
import torch  # noqa: F401 - imported for P-DQN compatibility

from .physics import MAX_POWER, ThermalPhysics
from .workload import ThunderWorkloadLoader, WorkloadGenerator


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


class LLNLThunderEnv(gym.Env):
    """Large-scale DC environment based on LLNL Thunder SWF trace (840 servers)."""

    metadata = {"render_modes": []}
    TOTAL_SERVERS = 840  # 20 racks * 42 nodes (used for heatmap reshaping)

    def __init__(
        self,
        workload_path: str | Path = "data/raw/LLNL-Thunder-2007-1.1-cln.swf",
        core_normalizer: float = 24.0,
        chunk_size: int = 8,
    ) -> None:
        super().__init__()

        self.physics = ThermalPhysics()
        self.alpha_heat = 0.005  # slower heating
        self.beta_cool = 0.2  # stronger cooling
        self.passive_cool = 0.05  # natural convection toward ambient
        self.workload = ThunderWorkloadLoader(
            str(workload_path), core_normalizer=core_normalizer, chunk_size=chunk_size
        )

        self.n_servers = self.TOTAL_SERVERS
        self.temps = np.full(self.n_servers, 25.0, dtype=np.float32)
        self.cpu_load = np.zeros(self.n_servers, dtype=np.float32)

        obs_dim = self.n_servers * 2 + 2  # loads + temps + job_req + runtime
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Tuple(
            (
                gym.spaces.Discrete(self.n_servers),
                gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            )
        )

        self.next_job_req, self.next_job_runtime = self.workload.step()

    def _get_obs(self) -> np.ndarray:
        temps_norm = np.clip(self.temps / 100.0, 0.0, 1.0)
        obs = np.concatenate(
            [
                self.cpu_load,
                temps_norm,
                np.array([self.next_job_req, self.next_job_runtime], dtype=np.float32),
            ]
        ).astype(np.float32)
        return obs

    def step(self, action: Tuple[int, Any]):
        server_idx, airflow = action
        airflow_value = float(np.array(airflow).reshape(-1)[0])
        airflow_value = float(np.clip(airflow_value, 0.0, 1.0))

        reward = 0.0

        # Task assignment with fallback to avoid dropping jobs.
        if self.cpu_load[server_idx] + self.next_job_req <= 1.0:
            target_idx = server_idx
        else:
            target_idx = int(np.argmin(self.cpu_load))
            reward -= 5.0  # scheduling error penalty

        self.cpu_load[target_idx] = float(
            np.clip(self.cpu_load[target_idx] + self.next_job_req, 0.0, 1.0)
        )

        it_power = 0.0
        for i in range(self.n_servers):
            server_power = self.physics.calculate_power(float(self.cpu_load[i]))
            temp = float(self.temps[i])
            ambient = float(self.physics.ambient_temp_c)
            temp_new = (
                temp
                + self.alpha_heat * server_power
                - self.beta_cool * airflow_value * temp
                - self.passive_cool * (temp - ambient)
            )
            self.temps[i] = temp_new
            it_power += server_power

        # Prevent runaway temperatures during simulation math.
        self.temps = np.clip(self.temps, 20.0, 100.0)

        cooling_power = airflow_value * MAX_POWER * self.n_servers
        total_power = it_power + cooling_power

        penalty = 1.0 if np.any(self.temps > 30.0) else 0.0
        reward = reward - total_power - 100.0 * penalty

        # Advance workload.
        self.next_job_req, self.next_job_runtime = self.workload.step()

        obs = self._get_obs()
        terminated = False
        truncated = False
        info = {
            "p_it_sum": it_power,
            "p_cooling": cooling_power,
            "penalty": penalty,
            "total_power": total_power,
        }
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.cpu_load = np.zeros(self.n_servers, dtype=np.float32)
        self.temps = np.full(self.n_servers, 25.0, dtype=np.float32)
        self.next_job_req, self.next_job_runtime = self.workload.step()
        return self._get_obs(), {}

    def render(self):  # pragma: no cover - not required
        return None
