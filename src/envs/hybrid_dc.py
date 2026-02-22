from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import gymnasium as gym
import numpy as np

from .workload import ThunderWorkloadLoader


class LLNLThunderEnv(gym.Env):
    """Vectorized LLNL Thunder data center environment for 840 servers."""

    metadata = {"render_modes": []}

    N_RACKS = 20
    SERVERS_PER_RACK = 42
    TOTAL_SERVERS = N_RACKS * SERVERS_PER_RACK
    MAX_CORES = 24
    P_IDLE = 100.0
    P_FULL = 300.0

    def __init__(
        self,
        workload_path: str | Path = "data/raw/LLNL-Thunder-2007-1.1-cln.swf",
        alpha: float = 0.05,
        beta: float = 0.5,
        cooling_scale: float = 0.05,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.cooling_scale = float(cooling_scale)
        self.ambient_temp = 25.0

        self.workload = ThunderWorkloadLoader(
            str(workload_path), core_normalizer=self.MAX_CORES, chunk_size=8
        )

        obs_low = np.concatenate(
            [
                np.array([0.0, 0.0], dtype=np.float32),
                np.zeros(self.TOTAL_SERVERS, dtype=np.float32),
                np.full(self.TOTAL_SERVERS, self.ambient_temp, dtype=np.float32),
            ]
        )
        obs_high = np.concatenate(
            [
                np.array([1.0, 1.0], dtype=np.float32),
                np.ones(self.TOTAL_SERVERS, dtype=np.float32),
                np.full(self.TOTAL_SERVERS, 100.0, dtype=np.float32),
            ]
        )
        self.observation_space = gym.spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )
        self.action_space = gym.spaces.Tuple(
            (
                gym.spaces.Discrete(self.TOTAL_SERVERS),
                gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            )
        )

        self.cpu_load = np.zeros(self.TOTAL_SERVERS, dtype=np.float32)
        self.temps = np.full(self.TOTAL_SERVERS, self.ambient_temp, dtype=np.float32)
        self.current_airflow = 0.0
        self.current_job_req, self.current_job_runtime = self.workload.step()

    def _get_obs(self) -> np.ndarray:
        return np.concatenate(
            [
                np.array(
                    [self.current_job_req, self.current_airflow], dtype=np.float32
                ),
                self.cpu_load,
                self.temps,
            ]
        ).astype(np.float32)

    def step(self, action: Tuple[int, Any]):
        server_idx, airflow = action
        airflow_value = float(np.array(airflow).reshape(-1)[0])
        airflow_value = float(np.clip(airflow_value, 0.0, 1.0))
        self.current_airflow = airflow_value

        job_req = float(self.current_job_req)
        reward = 0.0

        # Task assignment with capacity check.
        if self.cpu_load[server_idx] + job_req <= 1.0:
            self.cpu_load[server_idx] = float(self.cpu_load[server_idx] + job_req)
        else:
            reward -= 10.0

        # IT power model (vectorized).
        p_it = self.P_IDLE + (self.P_FULL - self.P_IDLE) * self.cpu_load

        # Thermal surrogate update (vectorized); airflow cools all servers equally.
        # Heat increases with IT power, cooling scales with airflow and power, plus ambient relaxation.
        self.temps = self.temps + self.alpha * p_it - self.beta * airflow_value * p_it
        self.temps = self.temps + 0.01 * (self.ambient_temp - self.temps)
        self.temps = np.clip(self.temps, self.ambient_temp, 100.0)

        # Cooling power scaled by airflow and fleet size.
        p_cooling = (
            airflow_value * self.P_FULL * self.TOTAL_SERVERS * self.cooling_scale
        )

        p_it_sum = float(np.sum(p_it))
        pue = (p_it_sum + p_cooling) / max(p_it_sum, 1e-6)

        penalty_t = float(np.mean(np.log1p(np.exp(self.temps - 30.0))))
        penalty_u = float(np.mean(np.log1p(np.exp(self.cpu_load - 0.9))))

        reward += -pue - penalty_t - penalty_u

        # Fetch next task (cores normalized by MAX_CORES, runtime normalized by max runtime).
        self.current_job_req, self.current_job_runtime = self.workload.step()

        obs = self._get_obs()
        terminated = False
        truncated = False
        info = {
            "p_it_sum": p_it_sum,
            "p_cooling": p_cooling,
            "pue": pue,
            "penalty_t": penalty_t,
            "penalty_u": penalty_u,
        }
        return obs, float(reward), terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.cpu_load = np.zeros(self.TOTAL_SERVERS, dtype=np.float32)
        self.temps = np.full(self.TOTAL_SERVERS, self.ambient_temp, dtype=np.float32)
        self.current_airflow = 0.0
        self.current_job_req, self.current_job_runtime = self.workload.step()
        return self._get_obs(), {}

    def render(self):  # pragma: no cover - not required
        return None
