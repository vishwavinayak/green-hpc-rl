

# 🌿 GreenHPC-RL: Intelligent Data Center Optimization

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Algorithm](https://img.shields.io/badge/Algorithm-P--DQN-green)
![Phase](https://img.shields.io/badge/Phase-2%20Research%20Scale-purple)
![MPS](https://img.shields.io/badge/Accelerator-Apple%20Silicon%20MPS-silver)

**Deep Reinforcement Learning for co-optimizing Job Scheduling and Cooling in Data Centers.**

---

## 🔬 Project Overview

Modern HPC data centers operate with a fundamental conflict:

*   **Job Schedulers** assign tasks to servers to maximise throughput — ignoring heat.
*   **Cooling Systems** react to thermal spikes to prevent hardware damage — ignoring workload.

These two systems work against each other, creating **thermal hotspots** and wasted electricity. **GreenHPC-RL** aims to address this conflict by training a single **Parameterized Deep Q-Network (P-DQN)** agent to control *both* layers simultaneously — choosing which server receives a job *and* setting the airflow rate — so that performance and energy efficiency are optimised jointly.

This repository presents a two-phase development process, from prototype validation to large-scale simulation.

---

## 🧪 Phase 1: Prototype — Google Borg Traces

**Goal:** Validate the P-DQN algorithm on a small, controlled cluster before scaling up.

| Property | Value |
| :--- | :--- |
| **Cluster Size** | 10 Servers |
| **Dataset** | Google Cluster Data 2011 (Borg Traces) |
| **Environment** | `src/envs/hybrid_dc.py` |
| **Episodes** | 500 |

### Result: Algorithm Validated — ~7% Energy Savings

The agent converged from a random policy to a stable energy-optimised policy, confirming that P-DQN can successfully navigate the hybrid (discrete + continuous) action space.

![Phase 1 Training Curve](logs/v1_google/learning_curve.png)

*   **Y-Axis:** Total Reward (Negative Energy Cost). Higher is better.
*   **Trend:** Reward rose from ~-355k (random) to ~-335k (optimised), representing a **~7% reduction** in cumulative energy cost relative to the Round-Robin baseline.

![Phase 1 AI vs Baseline](logs/v1_google/final_comparison.png)

*   **Top:** Energy Efficiency Comparison bar chart — GreenHPC-RL vs. Baseline cumulative power.
*   **Bottom:** Thermal heatmaps showing the difference in server temperature distribution between the two agents.

---

## 🚀 Phase 2: Research Scale — LLNL Thunder Supercomputer

**Goal:** Scale the validated algorithm to a HPC cluster of realistic size, using vectorised physics and hardware acceleration.

| Property | Value |
| :--- | :--- |
| **Cluster Size** | **840 Servers** (20 Racks × 42 Nodes) |
| **Dataset** | LLNL Thunder SWF Traces (real HPC job logs) |
| **Environment** | `src/envs/llnl_thunder.py` — Vectorised NumPy physics |
| **Accelerator** | Apple Silicon MPS (PyTorch) |

### Result: 39.0% Energy Savings — Emergent Strategy Observed

At research scale, the agent independently observed a **Sparse Load Distribution** strategy: rather than filling servers sequentially like Round-Robin, it deliberately spreads jobs across racks to keep per-server temperatures low, enabling the cooling system to run at a fraction of full power.

![Phase 2 Comparison](logs/v2_thunder/thunder_comparison.png)

*   **Top:** Energy bar chart — GreenHPC-RL vs. Baseline average power draw.
*   **Bottom-Left (Baseline):** Viridis heatmap shows concentrated load (bright yellow hotspots) caused by Round-Robin packing.
*   **Bottom-Right (GreenHPC-RL):** Uniform dark-green distribution — the agent spreads CPU utilisation evenly across all 20 racks.
*   **Key Stat:** **39.0% Energy Savings** compared to the industry-standard Round-Robin baseline.

---

## 🛠️ Tech Stack

| Component | Technology |
| :--- | :--- |
| **RL Algorithm** | Parameterized DQN (P-DQN), PyTorch |
| **Environment** | Gymnasium, NumPy (fully vectorised physics) |
| **Acceleration** | Apple Silicon MPS via `torch.backends.mps` |
| **Workload Data** | Google Borg 2011 · LLNL Thunder SWF |
| **Visualisation** | Matplotlib, Seaborn |

### Algorithm: Hybrid Action Space (P-DQN)

Standard RL algorithms (DQN, PPO) cannot handle simultaneous discrete and continuous control. P-DQN separates the two:

*   **Discrete Action:** Select a server $k \in \{1 \ldots N\}$ for the incoming job.
*   **Continuous Parameter:** Set the Airflow Rate $x_k \in [0, 1]$ linked to that choice.

The agent minimises a hybrid loss:

$$\mathcal{L} = \mathcal{L}_{Q}(\text{MSE}) + \mathcal{L}_{Actor}(-Q_{val})$$

---

## � Repository Structure

```text
green-hpc-rl/
├── configs/
│   └── default.yaml               # Hyperparameters
├── data/
│   └── raw/
│       └── borg_traces_data.csv   # Phase 1 workload traces
├── logs/
│   ├── v1_google/                 # Phase 1 outputs (learning curve, model)
│   └── v2_thunder/                # Phase 2 outputs (benchmark, heatmaps)
├── scripts/
│   ├── train.py                   # Phase 1 training loop
│   ├── plot_results.py            # Phase 1 learning curve
│   ├── evaluate.py                # Phase 2 AI vs Baseline benchmark
│   └── plot_thunder.py            # Phase 2 comparison visualisation
├── src/
│   ├── agents/
│   │   ├── pdqn_agent.py          # P-DQN Agent (PyTorch)
│   │   ├── networks.py            # Actor / Q-Network architectures
│   │   ├── buffer.py              # Hybrid Replay Buffer
│   │   └── baseline.py            # Round-Robin + Reactive baseline
│   ├── envs/
│   │   ├── hybrid_dc.py           # Phase 1 Gymnasium environment
│   │   ├── physics.py             # Thermodynamic equations
│   │   └── workload.py            # Google Borg trace parser
│   └── utils/
│       ├── logger.py              # CSV / checkpoint logging
│       └── plotting.py            # Shared plot helpers
├── tests/
├── pyproject.toml
└── README.md
```

---

## �💻 Usage

This project uses **uv** for dependency management.

```bash
# 1. Clone and install
git clone https://github.com/your-username/green-hpc-rl.git
cd green-hpc-rl
uv sync
```

### Phase 1 — Google Borg Prototype
```bash
# Train
uv run python scripts/train.py

# Plot learning curve
uv run python scripts/plot_results.py
# → logs/v1_google/learning_curve.png
```

### Phase 2 — LLNL Thunder Scale
```bash
# Train on Thunder environment
uv run python scripts/train_thunder.py

# Run AI vs Baseline benchmark
uv run python scripts/evaluate.py

# Generate comparison plots
uv run python scripts/plot_thunder.py
# → logs/v2_thunder/thunder_comparison.png
```

---

## 📚 References

1.  **Framework:** Ran, Y. et al., *"Optimizing Energy Efficiency for Data Center via Parameterized Deep Reinforcement Learning"*, IEEE Transactions on Services Computing.
2.  **Algorithm:** Xiong, J. et al., *"Parametrized Deep Q-Networks Learning"*, arXiv:1810.06394.
3.  **Dataset (Phase 1):** Google Cluster Data (Borg), 2011.
4.  **Dataset (Phase 2):** LLNL Thunder Workload Traces, Parallel Workloads Archive.

---
