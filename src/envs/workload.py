import ast
from collections import deque
from typing import Deque, List, Tuple

import pandas as pd


class WorkloadGenerator:
    """Simple iterator over normalized CPU load values from a Borg trace."""

    def __init__(self, csv_path: str) -> None:
        self._df = pd.read_csv(csv_path)
        # Drop missing usage entries and parse the stored dict strings.
        self._df = self._df.dropna(subset=["average_usage"])
        self._df["average_usage"] = self._df["average_usage"].apply(ast.literal_eval)
        self._df["cpu_load"] = self._df["average_usage"].apply(self._extract_cpu)

        max_cpu = self._df["cpu_load"].max()
        if pd.notna(max_cpu) and max_cpu > 1.0:
            self._df["cpu_load"] = self._df["cpu_load"] / max_cpu

        self._cpu_values = self._df["cpu_load"].astype(float).reset_index(drop=True)
        self._index = 0

    @staticmethod
    def _extract_cpu(usage: dict) -> float:
        return float(usage.get("cpus", 0.0))

    def step(self) -> float:
        if self._cpu_values.empty:
            raise ValueError("No workload data available after preprocessing.")

        value = float(self._cpu_values.iloc[self._index])
        self._index = (self._index + 1) % len(self._cpu_values)
        return value

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._cpu_values)


class ThunderWorkloadLoader:
    """Parses the LLNL Thunder SWF trace and yields normalized tasks."""

    def __init__(
        self, swf_path: str, core_normalizer: float = 24.0, chunk_size: int = 8
    ) -> None:
        if core_normalizer <= 0:
            raise ValueError("core_normalizer must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        self._core_normalizer = float(core_normalizer)
        self._chunk_size = chunk_size
        self._tasks_template, self._max_runtime = self._parse_trace(swf_path)
        if not self._tasks_template:
            raise ValueError("No tasks parsed from workload file.")
        self._queue: Deque[Tuple[int, float]] = deque(self._tasks_template)

    def _parse_trace(self, swf_path: str) -> Tuple[List[Tuple[int, float]], float]:
        tasks: List[Tuple[int, float]] = []
        max_runtime = 0.0

        with open(swf_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith(";"):
                    continue

                parts = line.split()
                if len(parts) <= 4:
                    continue

                try:
                    runtime = float(parts[3])
                    req = int(parts[4])
                except ValueError:
                    continue

                if runtime <= 0 or req <= 0:
                    continue

                max_runtime = max(max_runtime, runtime)
                tasks.extend(self._split_job(req, runtime))

        if max_runtime <= 0:
            max_runtime = 1.0

        return tasks, max_runtime

    def _split_job(self, req: int, runtime: float) -> List[Tuple[int, float]]:
        if req <= 24:
            return [(req, runtime)]

        subtasks: List[Tuple[int, float]] = []
        full_chunks = req // self._chunk_size
        remainder = req % self._chunk_size

        for _ in range(full_chunks):
            subtasks.append((self._chunk_size, runtime))
        if remainder:
            subtasks.append((remainder, runtime))

        return subtasks

    def step(self) -> Tuple[float, float]:
        if not self._queue:
            self._queue = deque(self._tasks_template)

        cores, runtime = self._queue.popleft()
        cores_norm = float(cores) / self._core_normalizer
        runtime_norm = float(runtime) / self._max_runtime

        return cores_norm, runtime_norm

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._queue)
