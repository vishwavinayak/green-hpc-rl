from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.workload import WorkloadGenerator

loader = WorkloadGenerator(ROOT / "data/raw/borg_traces_data.csv")
print("Dataset Size:", len(loader))
print("First 5 loads:", [loader.step() for _ in range(5)])
