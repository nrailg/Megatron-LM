# Stronger-MAS: Multi-Agent Reinforcement Learning for Collaborative LLMs

## Paper Analysis: Contributions & Core Mathematical Formulas

**Paper:** arXiv:2510.11062v5 (Yujie Zhao et al., UCSD & Intel)

---

## 1. Core Problem

The paper addresses the gap between two popular paradigms for improving LLM agents:

- **Multi-Agent Systems (MAS):** Enhance performance through role-based orchestration and inter-agent collaboration (e.g., a Coder + Tester loop), but rely on *prompt-only* augmentation without weight updates.
- **Reinforcement Learning (RL):** Train stronger policies via environment rewards (e.g., GRPO), but are designed for *single-agent* settings.

**Key insight:** Standard GRPO's grouping assumption—that all candidates in a comparison group share the same prompt—breaks down in MAS because prompts vary by *role* (different agents) and by *turn* (interaction history accumulates). Naively applying parallel sampling yields group-size-1 at later turns, destroying the variance-reduction effect.

---

## 2. Main Contributions

### Contribution 1: AT-GRPO Algorithm

An **Agent- and Turn-wise Grouped RL** algorithm consisting of three intertwined ideas:

| Component | What it solves |
|---|---|
| **Tree-structured Sampling** | At each turn, branch K candidates per agent from the *same state*, so all K share an identical prompt. Greedy selection of the best-rewarded action propagates the rollout forward. |
| **Agent- & Turn-wise Grouping** | Group key = `hash(environment, agent, turn)`. Only candidates with the same role, turn, and environment are compared, ensuring prompt identity for valid advantage estimation. |
| **Agent-wise Credit Assignment** | Mixed reward combining a global team reward and agent-local subtask reward, balancing collective and role-specific learning signals. |

### Contribution 2: MAS Training System

A system-level design enabling on-policy RL for MAS:

- **Per-model GPU Resource Pools:** Each policy gets its own RolloutWorker + UpdateWorker, enabling concurrent multi-policy training.
- **CPU Environment Pool:** Fleet of sandboxed EnvWorkers supporting thousands of parallel rollouts.
- **Router:** Dispatches trajectories to the correct UpdateWorker based on policy assignment σ(i).
- Supports both **role-sharing** (M=1, single policy) and **role-specialized** (M=N, one policy per agent) regimes.

### Contribution 3: Extensive Empirical Validation

Evaluated on Qwen3 (1.7B and 8B) across four diverse domains:

| Domain | Improvement over single-agent RL baseline |
|---|---|
| **Long-horizon Planning** (Sokoban, Plan-Path) | 14.0–47.0% → **96.0–99.5%** accuracy |
| **Coding** (LiveCodeBench, APPS, CodeContests) | Average gain of **3.87–7.62%** |
| **Math** (AIME24, AIME25, OlympiadBench) | Average gain of **9.0–17.93%** |
| **Game** (Sudoku) | Near-perfect (99%) accuracy |

### Contribution 4: Analytical Insights

- RL training on MAS **reinforces role-specific specialization** (measured via response similarity divergence between roles).
- The choice between role-sharing vs. role-specialized policies is **task-dependent**: structured tasks (game/planning) favor shared policies; open-ended tasks (coding) benefit from specialization.

---

## 3. Core Mathematical Formulas

### 3.1 Markov Game Formulation (MAS Setting)

The N-agent LLM system is modeled as a Markov game:

$$\mathcal{M} = (\mathcal{S}, \{\mathcal{A}_i\}_{i=1}^{N}, \mathcal{T}, \{r_i\}_{i=1}^{N}, T, H)$$

where:
- $\mathcal{S}$: state space
- $\mathcal{A}_i$: action space of agent $i$
- $\mathcal{T}$: transition function inducing intra-turn micro-transitions: $s_{t,0} = s_t$ and $s_{t,i} = \mathcal{T}(s_{t,i-1}, a_{t,i}, i)$, culminating in $s_{t+1} = s_{t,N}$
- $r_i: \mathcal{A}_i \to [0, 1]$: reward function for agent $i$
- $T$: turn horizon; $H$: optimization step horizon
- $\sigma: \{1, \ldots, N\} \to \{1, \ldots, M\}$: maps each agent to an LLM policy
- Each agent $i$ observes $o_{t,i} = o_i(s_t, h_t)$ (state + interaction history)

### 3.2 Group-Relative Advantage (Eq. 1 — Baseline GRPO)

For K candidate actions $\{a_t^{(c)}\}_{c=1}^{K}$ sampled from the same prompt, the group-relative advantage is:

$$A_g\big(a_t^{(c)}\big) = \frac{R(a_t^{(c)}) - \text{mean}\left(\{R(a_t^{(c)})\}_{c=1}^{K}\right)}{F_{\text{norm}}\left(\{R(a_t^{(c)})\}_{c=1}^{K}\right)}$$

**Meaning:** The advantage of each candidate is its reward centered by the group mean and normalized by a dispersion measure $F_{\text{norm}}$ (e.g., standard deviation). This is the standard GRPO advantage that AT-GRPO adapts.

**Key problem in MAS:** In multi-agent multi-turn settings, parallel sampling produces group-size-1 beyond turn 1 (since no two trajectories share the same prompt), collapsing the normalization and eliminating variance reduction.

### 3.3 Clipped Policy Optimization Loss (Eq. 2 — PPO-style with GRPO advantages)

The per-model policy loss using clipped importance-weighted advantages:

$$\mathcal{L}(\theta^{(m)}) = -\mathbb{E}_{g \in \mathcal{B}_m} \left[ \frac{1}{K} \sum_{c=1}^{K} \min\left( r_g^{(c,m)}(\theta^{(m)}) A_g^{(c)},\ \text{clip}\left(r_g^{(c,m)}(\theta^{(m)}),\ 1-\varepsilon,\ 1+\varepsilon\right) A_g^{(c)} \right) \right]$$

where:
- $r(\theta) = \frac{\pi_\theta(o_i | q)}{\pi_{\theta_{\text{old}}}(o_i | q)}$ is the importance sampling ratio (new policy / old policy)
- $\varepsilon$ is the PPO clipping parameter
- $\mathcal{B}_m = \bigcup_{i : \sigma(i) = m} \mathcal{D}_i$ is the per-model training batch

**Two optimization regimes:**
- **Role-sharing** ($M = 1$): $\mathcal{B}_1 = \bigcup_{i=1}^{N} \mathcal{D}_i$, single joint update $\theta^1 \leftarrow \theta^1 - \eta \nabla_{\theta^1} \mathcal{L}(\theta^1)$
- **Role-specialized** ($M = N$): $\mathcal{B}_i = \mathcal{D}_i$, independent per-agent updates $\theta^{(i)} \leftarrow \theta^{(i)} - \eta \nabla_{\theta^{(i)}} \mathcal{L}(\theta^{(i)})$

### 3.4 Agent-wise Credit Assignment / Mixed Reward (Eq. 3 — Key novelty)

The final reward for agent $i$ at turn $t$ is a mixture of global and local components:

$$r_{t,i} = \alpha \cdot r_t^{\text{team}} + r_{t,i}^{\text{loc}}$$

where:
- $r_t^{\text{team}}$: global team reward (e.g., pass rate of generated code on golden unit tests)
- $r_{t,i}^{\text{loc}}$: agent-specific local reward evaluating subtask quality (e.g., coder's code pass rate, tester's unit test quality)
- $\alpha$: hyperparameter balancing team vs. individual incentives (set to 1 in experiments)

**Example (Coder-Tester MAS for code tasks):**
- $r^{\text{team}}$ = pass rate of generated program on golden unit tests
- $r_{\text{coder}}^{\text{loc}}$ = coder's code pass rate
- $r_{\text{tester}}^{\text{loc}}$ = pass rate of golden reference solution against the tester's generated tests

### 3.5 AT-GRPO Group Key (Algorithm 1, line 8)

The agent- and turn-wise group key that ensures valid GRPO comparisons:

$$g \leftarrow \text{hash}(e, i, t)$$

where $e$ is the environment instance, $i$ is the agent index, and $t$ is the turn number. This guarantees that only candidates sharing the same environment, agent role, and turn (thus identical prompt) are grouped together for advantage computation.

### 3.6 Tree-structured Greedy Selection (Algorithm 1, line 10-11)

At each (turn, agent) node, after sampling K branches and computing advantages:

$$c^\star \leftarrow \arg\max_c\ r_{t,i,e}^{(c)}$$
$$a_{t,i,e} \leftarrow a_{t,i,e}^{(c^\star)}$$

The best-rewarded candidate is selected to continue the trajectory. This **greedy tree pruning** strategy:
1. Ensures subsequent turns see a high-quality state (concentrating exploration on coordination-critical decisions)
2. Maintains a balanced mix of positive/negative advantage samples across the K branches
3. Stabilizes training optimization

---

## 4. How the Formulas Connect (End-to-End Flow)

```
For each training step:
  │
  ├─ Phase 1: On-Policy Rollout
  │    For each environment e, turn t, agent i:
  │      1. Sample K candidate actions from π_θ(σ(i))
  │      2. Compute mixed reward r_{t,i} for each candidate     ← Eq. 3
  │      3. Define group key g = hash(e, i, t)                   ← AT grouping
  │      4. Compute group-relative advantage A_g^(c)             ← Eq. 1
  │      5. Greedy select best candidate for next state          ← Tree sampling
  │      6. Store (g, observation, K actions, K advantages) in D_i
  │
  └─ Phase 2: Policy Update
       For each model m:
         1. Construct batch B_m from all agents assigned to model m
         2. Compute clipped loss L(θ^(m))                        ← Eq. 2
         3. Update θ^(m) via gradient descent
```

---

## 5. Key Differences from Prior Work

| Aspect | Standard GRPO / Prior MAS-RL | AT-GRPO (this paper) |
|---|---|---|
| **Grouping** | Same-question parallel samples | Agent- and turn-wise: hash(env, agent, turn) |
| **Sampling** | Parallel full trajectories | Tree-structured: branch K at each turn node |
| **Reward** | Single scalar | Mixed: α·r_team + r_loc_i |
| **Policy regime** | Single-agent only | Both role-sharing (M=1) and role-specialized (M=N) |
| **Domains** | Mostly math-only | Game, planning, coding, and math |
| **System** | Single model pool | Per-model GPU pools + CPU env pool + router |

---

## 6. Limitations (as acknowledged by authors)

1. Scalability beyond 2 agents not yet explored (all experiments use N=2).
2. MAS workflow design is task-specific and currently hand-crafted.
3. The reward mixing coefficient α=1 is not tuned; better task-specific tuning may yield further gains.
4. Role-specialized training doubles GPU memory requirements.
