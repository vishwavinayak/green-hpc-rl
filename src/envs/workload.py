import ast

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
