# Sample Based Learning Methods in Reinforcement Learning

## Monte Carlo Methods

### General Idea
Monte Carlo (MC) methods learn from complete episodes of experience without requiring prior knowledge of environment dynamics. They use the average of sample returns (complete episodes) to estimate value functions and discover optimal policies.

### Estimating Value Functions with MC
- Uses complete episode returns to estimate value functions
- Value estimate for a state = average of returns following all visits to that state
- Requires waiting until episode completion to update values
- Formula: $V(s) = \frac{\sum_{i=1}^{N} G_i(s)}{N}$ where $G_i(s)$ is the return following the $i$-th visit to state $s$

### Problems Solvable via MC Methods
- Episodic tasks only (requires termination)
- Does not require complete environment dynamics knowledge
- Works well for high-dimensional and complex problems
- Effective for problems with delayed rewards

### MC Prediction
First-visit MC: Average returns following the first visit to a state in each episode.
Every-visit MC: Average returns following every visit to a state.

Algorithm:
```
Initialize V(s) arbitrarily for all s ∈ S
Initialize returns(s) as empty list for all s ∈ S

For each episode:
    Generate an episode following π: S₀, A₀, R₁, S₁, A₁, ..., Sₜ
    G ← 0
    For each step t = T-1, T-2, ..., 0:
        G ← γG + Rₜ₊₁
        If Sₜ not in S₀, S₁, ..., Sₜ₋₁ (for first-visit):
            Append G to returns(Sₜ)
            V(Sₜ) ← average(returns(Sₜ))
```

### Exploration in MC
- Critical for convergence to optimal policy
- Must visit all state-action pairs to learn their values
- Cannot rely only on greedy policies (exploitation)

### MC for Generalized Policy Iteration (GPI)
- Policy evaluation: Estimate $V_π$ or $Q_π$ using MC returns
- Policy improvement: Make policy greedy with respect to current value function
- Iterate until convergence

### Exploring Starts
- Ensures all state-action pairs are tried by starting episodes at random state-action pairs
- Problematic in real-life because:
  1. Cannot control environment's starting state
  2. Impossible in online learning scenarios
  3. Impractical for large state spaces

### On-Policy vs Off-Policy
- **On-policy**: Learn about the policy being executed
- **Off-policy**: Learn about a policy different from the one being executed (e.g., learn optimal policy while following exploratory policy)

### On-Policy First-Visit MC Control
```
Initialize Q(s,a) arbitrarily for all s ∈ S, a ∈ A
Initialize returns(s,a) as empty list for all s ∈ S, a ∈ A
Initialize policy π(s) arbitrarily for all s ∈ S

Repeat forever:
    Generate an episode following π: S₀, A₀, R₁, S₁, A₁, ..., Sₜ
    G ← 0
    For each step t = T-1, T-2, ..., 0:
        G ← γG + Rₜ₊₁
        Unless the pair Sₜ, Aₜ appears in S₀,A₀, S₁,A₁, ..., Sₜ₋₁,Aₜ₋₁:
            Append G to returns(Sₜ, Aₜ)
            Q(Sₜ, Aₜ) ← average(returns(Sₜ, Aₜ))
            π(Sₜ) ← argmax_a Q(Sₜ, a)
```

### ε-greedy Policy for On-Policy MC
```
π(a|s) = {
    1-ε+ε/|A(s)| if a = argmax_a Q(s,a)
    ε/|A(s)|     for all other a
}
```

### Importance Sampling
- Technique for estimating expected values under one distribution using samples from another
- Corrects for the difference between target and behavior policies
- Ordinary importance sampling: $V(s) = \frac{\sum_{i=1}^{N} \rho_i G_i}{\sum_{i=1}^{N} \rho_i}$
- Weighted importance sampling: $V(s) = \frac{\sum_{i=1}^{N} \rho_i G_i}{N}$
- Where $\rho_i = \frac{\prod_{t=0}^{T-1} \pi(A_t|S_t)}{\prod_{t=0}^{T-1} b(A_t|S_t)}$ is the importance sampling ratio

## Temporal Difference Methods

### General Idea
Temporal Difference (TD) methods learn from incomplete episodes by bootstrapping (using estimates of future values). They combine ideas from Monte Carlo and Dynamic Programming.

### TD Error
The difference between the estimated return and the current value estimate:
$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$

### TD(0) Algorithm
```
Initialize V(s) arbitrarily for all s ∈ S
Initialize π to the policy to be evaluated

For each episode:
    Initialize S
    For each step of episode:
        Take action A according to π, observe R, S'
        V(S) ← V(S) + α[R + γV(S') - V(S)]
        S ← S'
    Until S is terminal
```

### Benefits of Learning Online with TD
1. Learn without knowing the environment model (unlike DP)
2. Update estimates on each step without waiting for episode end (unlike MC)
3. Lower variance than MC (but potentially more bias)
4. Work in continuing environments (unlike MC)
5. More computationally efficient

### Advantages Over DP and MC
- vs DP: No need for complete environment model
- vs MC: Can learn from incomplete sequences and in continuing tasks
- vs MC: Usually more efficient in stochastic environments
- vs Both: Often converges faster in practice

### TD for Control/GPI
Similar to MC, but using TD methods for policy evaluation:
1. Policy evaluation: Use TD to estimate value function
2. Policy improvement: Make policy greedy with respect to value function
3. Iterate until convergence

### Sarsa Control Algorithm
On-policy TD control algorithm:
```
Initialize Q(s,a) arbitrarily for all s ∈ S, a ∈ A
Initialize π to be ε-greedy with respect to Q

For each episode:
    Initialize S
    Choose A from S using π
    For each step of episode:
        Take action A, observe R, S'
        Choose A' from S' using π
        Q(S,A) ← Q(S,A) + α[R + γQ(S',A') - Q(S,A)]
        S ← S'; A ← A'
    Until S is terminal
```

### Performance Analysis of Learning Algorithms
- Sample efficiency: How much experience is needed to reach good performance
- Computational efficiency: Required computation per step
- Parameter sensitivity: How algorithm performance depends on parameters
- Asymptotic performance: Performance after learning converges
- Convergence guarantees: Does it provably converge to optimal solution
- Online vs. batch performance: How well algorithm performs during learning

### Q-learning Algorithm
Off-policy TD control algorithm:
```
Initialize Q(s,a) arbitrarily for all s ∈ S, a ∈ A

For each episode:
    Initialize S
    For each step of episode:
        Choose A from S using policy derived from Q (e.g., ε-greedy)
        Take action A, observe R, S'
        Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]
        S ← S'
    Until S is terminal
```

### Relationship with Bellman Optimality Equations
Q-learning directly approximates the optimal action-value function $q_{*}$, independent of the policy being followed. It uses the Bellman optimality equation as an update rule:
$Q(S,A) ← Q(S,A) + α[R + γ \max_a Q(S',a) - Q(S,A)]$

### On-Policy Sarsa vs Off-Policy Q-learning
- Sarsa considers the actual next action (safer, more conservative)
- Q-learning considers the best next action (optimistic, riskier in some cases)
- Q-learning can learn optimal policy while following exploratory policy
- Sarsa finds a policy that balances exploration and exploitation
- Q-learning generally more sample efficient but less stable
- Sarsa tends to perform better in practice with function approximation

### Expected Sarsa
Instead of using the value of the next selected action, uses the expected value of the next state considering the policy:
$Q(S,A) ← Q(S,A) + α[R + γ \sum_a \pi(a|S')Q(S',a) - Q(S,A)]$

### Expected Sarsa vs Sarsa vs Q-learning
- Expected Sarsa eliminates the variance due to random action selection in Sarsa
- Expected Sarsa reduces to Q-learning if target policy is greedy
- Expected Sarsa has lower variance than Sarsa (more stable learning)
- Expected Sarsa is more computationally expensive per step
- Order of performance (usually): Expected Sarsa > Q-learning > Sarsa

## Planning

### Planning
Using a model of the environment to consider possible future situations and decide on a course of action.

### Learning
Improving the value function or policy based on experience.

### Acting
Selecting actions based on the current policy to interact with the environment.

### Model
A representation of the environment's dynamics that allows simulation of experience.
- **Transition model**: $p(s'|s,a)$ - State transition probabilities
- **Reward model**: $r(s,a,s')$ - Expected rewards

### Distribution vs Sample Models
- **Distribution model**: Complete probability distributions for transitions and rewards
- **Sample model**: Generates samples from transitions and reward distributions

#### When to use:
- **Distribution model**: When transition probabilities are known and manageable
- **Sample model**: When environment is complex and sampling is easier than computing distributions

#### Advantages vs Disadvantages:
- **Distribution model**:
  - ✓ Can compute expectations exactly
  - ✓ Better performance with smaller number of updates
  - ✗ Higher computational cost per update
  - ✗ May be impractical for large state spaces

- **Sample model**:
  - ✓ Lower computational cost per update
  - ✓ Scales better to large state spaces
  - ✓ Easier to implement in many domains
  - ✗ May require more updates for convergence
  - ✗ Introduces sampling variance

### Random-Sample One-Step Tabular Q-learning
```
Loop forever:
    S ← random state
    A ← random action
    Use model to compute R, S' (reward and next state)
    Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]
```

### Direct RL vs Planning Updates
- **Direct RL update**: Uses real experience from environment interaction
- **Planning update**: Uses simulated experience from model
- Both can use same algorithms (e.g., Q-learning, Sarsa)
- Planning can be done while interacting with environment (background planning)

### Dyna Architecture
Integrates acting, planning, and learning:
1. Collect real experience from environment
2. Learn/update model from real experience
3. Update value function / policy from real experience (direct RL)
4. Simulate experience using model
5. Update value function / policy from simulated experience (planning)

### Tabular Dyna-Q
```
Initialize Q(s,a) and Model(s,a) for all s ∈ S, a ∈ A
For each episode:
    S ← current state
    A ← ε-greedy(S, Q)
    Take action A, observe reward R and next state S'
    Q(S,A) ← Q(S,A) + α[R + γ max_a Q(S',a) - Q(S,A)]  // Direct RL
    Model(S,A) ← R, S'  // Model learning
    
    // Planning (n simulated steps)
    For i = 1 to n:
        S_sim ← random previously observed state
        A_sim ← random previously taken action in S_sim
        R_sim, S'_sim ← Model(S_sim, A_sim)
        Q(S_sim, A_sim) ← Q(S_sim, A_sim) + α[R_sim + γ max_a Q(S'_sim, a) - Q(S_sim, A_sim)]

    S ← S'
```

### Model Inaccuracies and Effects on Planning
- **Types of inaccuracies**:
  1. Incorrect transitions (wrong next states)
  2. Incorrect rewards
  3. Incomplete (missing states/actions)
  4. Oversimplified (ignoring stochasticity)

- **Effects on planning**:
  1. Suboptimal policies
  2. Confident but wrong value estimates
  3. Limited exploration
  4. Inability to adapt to environment changes

### Dyna-Q+ Algorithm
Extends Dyna-Q to encourage exploration of states/actions not recently tried:
```
// Changes to standard Dyna-Q:
Initialize τ(s,a) = 0 for all s, a (time since last visit)

For each step:
    // Update visit times for all state-actions
    For all s, a: τ(s,a) ← τ(s,a) + 1
    
    // Reset timer for current state-action
    τ(S,A) ← 0
    
    // In planning phase, modify rewards to encourage exploration
    R_sim, S'_sim ← Model(S_sim, A_sim)
    // Add exploration bonus based on time since last visit
    R_sim ← R_sim + κ√τ(S_sim, A_sim)
```
Where κ is an exploration bonus parameter that controls the degree of exploration.
