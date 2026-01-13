from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.envs.hybrid_dc import HybridDataCenterEnv


env = HybridDataCenterEnv(ROOT / "data/raw/borg_traces_data.csv")
obs, _ = env.reset()
print("Obs shape:", obs.shape)

# Test a sample action: Server 0, Airflow 0.5
action = (0, [0.5])
obs, reward, terminated, truncated, info = env.step(action)
print("Reward:", reward)
print("Terminated:", terminated, "Truncated:", truncated)
print("Info:", info)
