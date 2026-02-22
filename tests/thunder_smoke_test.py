import sys
from pathlib import Path

import numpy as np
import torch

# Ensure project root is on PYTHONPATH when running as a script.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.pdqn_agent import PDQNAgent
from src.envs.hybrid_dc import LLNLThunderEnv


def main() -> None:
    env = LLNLThunderEnv(workload_path=Path("data/raw/LLNL-Thunder-2007-1.1-cln.swf"))
    print(f"Observation space shape: {env.observation_space.shape}")

    state_size = int(env.observation_space.shape[0])
    n_servers = int(env.TOTAL_SERVERS)
    action_param_size = int(env.action_space[1].shape[0])

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    agent = PDQNAgent(
        state_size,
        n_servers,
        action_param_size,
        batch_size=4,
        device=device,
    )
    print(f"Agent device: {agent.device}")

    state, _ = env.reset()

    for step_idx in range(1, 21):
        server_choice = int(np.random.randint(0, n_servers))
        airflow = float(np.random.rand())
        action = (server_choice, np.array([airflow], dtype=np.float32))

        state, reward, terminated, truncated, _ = env.step(action)

        avg_temp = float(np.mean(env.temps))
        max_temp = float(np.max(env.temps))

        print(
            f"Step {step_idx:02d} | Server {server_choice:03d} | "
            f"Avg Temp {avg_temp:.2f} | Max Temp {max_temp:.2f} | Reward {reward:.2f}"
        )

        if terminated or truncated:
            break


if __name__ == "__main__":
    main()
