from __future__ import annotations

from typing import Tuple

import numpy as np


class BaselineAgent:
    """Rule-based baseline: round-robin scheduling with simple thermal reaction."""

    def __init__(self, action_dim: int = 840) -> None:
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        self.action_dim = action_dim
        self._counter = 0

    def select_action(self, state: np.ndarray) -> Tuple[int, list[float]]:
        # Round-robin server selection.
        server_index = self._counter % self.action_dim
        self._counter += 1

        # Temperatures live in indices 842:1682 for the 840-server env.
        temps = state[842:1682]
        max_temp = float(np.max(temps)) if temps.size else 0.0

        airflow = 0.9 if max_temp > 28.0 else 0.3
        return server_index, [airflow]
