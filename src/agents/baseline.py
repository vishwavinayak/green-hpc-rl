from typing import Sequence, Tuple

class BaselineAgent:
    """Rule-based baseline using round-robin scheduling and reactive airflow."""

    def __init__(self, n_servers: int) -> None:
        if n_servers <= 0:
            raise ValueError("n_servers must be positive")
        self.n_servers = n_servers
        self._counter = 0

    def select_action(self, state: Sequence[float]) -> Tuple[int, list[float]]:
        if len(state) < 20:
            raise ValueError(
                "state must contain at least 20 elements for temperature parsing"
            )

        server_index = self._counter % self.n_servers
        self._counter += 1

        temps = state[10 : 10 + self.n_servers]
        airflow = 0.9 if any(t > 25.0 for t in temps) else 0.3

        return server_index, [airflow]
