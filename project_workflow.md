# Project Workflow: Green HPC with Parameterized DQN

## Intro
This project explores hybrid control for data center operations using a Parameterized Deep Q-Network (P-DQN). The goal is to jointly decide (1) which server handles incoming jobs (discrete choice) and (2) the cooling airflow level (continuous parameter), minimizing energy while respecting thermal constraints. A custom environment models workloads, power, and temperature dynamics, and a PDQN agent learns from simulated experience.

## Methodology
1. **Environment Modeling**
   - *Hybrid action space*: server index (discrete) + airflow setting (continuous).
   - *State representation*: server loads, normalized temperatures, next job size.
   - *Dynamics*: thermal physics updates temperatures given IT power and airflow; workload generator supplies arriving jobs; reward penalizes power and overheating.
2. **Agent Architecture (PDQN)**
   - *ActorParamNetwork*: outputs continuous airflow parameters conditioned on state.
   - *QNetwork*: estimates Q-values for each discrete server given state + actor-produced parameters.
   - *Target networks*: stabilized copies for both actor and critic.
3. **Learning Loop**
   - *Experience collection*: epsilon-greedy action selection; environment step produces next state, reward, done.
   - *Replay buffer*: `HybridReplayBuffer` stores transitions for off-policy updates.
   - *Critic update*: MSE between current Q for chosen server and bootstrapped target using target networks.
   - *Actor update*: maximize expected Q via gradient ascent on actor parameters.
   - *Polyak averaging*: soft update of target networks with small τ.
4. **Training Schedule**
   - Episodes with capped steps; epsilon decays each episode.
   - Periodic model checkpoints; reward logging to CSV for visualization.
5. **Evaluation & Visualization**
   - Plot raw rewards and moving averages to assess convergence.
   - Inspect power/temperature trends and policy behaviors.

## Key Concepts
- **Hybrid RL (Parametric Actions)**: blending discrete choices with continuous parameters in a single policy/Q formulation.
- **Off-Policy Learning with Replay**: decorrelates samples, improves stability.
- **Target Networks & Polyak Averaging**: reduce estimation variance and oscillations.
- **Epsilon-Greedy Exploration**: balances exploration of servers and airflow settings.
- **Thermal-Power Tradeoff**: reward encodes both energy consumption and overheating penalties.

## Workflow Flowchart
```mermaid
flowchart TD
    A[Start Episode] --> B[Reset Env -> state]
    B --> C[Select action
(PDQN actor + epsilon-greedy)]
    C --> D[Env step -> next_state,
reward, done]
    D --> E[Store transition in
HybridReplayBuffer]
    E --> F[Update agent
(critic & actor)]
    F --> G[Soft update targets]
    G --> H{done or max steps?}
    H -- No --> C
    H -- Yes --> I[Log reward,
update epsilon]
    I --> J{Episode < max?}
    J -- Yes --> B
    J -- No --> K[Save metrics & model,
plot learning curve]
    K --> L[End]
```

## File Pointers
- Training loop: `scripts/train.py`
- Agent: `src/agents/pdqn_agent.py`
- Networks: `src/agents/networks.py`
- Replay buffer: `src/agents/buffer.py`
- Environment: `src/envs/hybrid_dc.py`
- Plotting: `scripts/plot_results.py`
