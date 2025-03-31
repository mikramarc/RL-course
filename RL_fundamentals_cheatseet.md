
# Reinforcement Learning Fundamentals

## Core Concepts

### Agent

An entity that takes actions in an environment to maximize cumulative reward. The agent observes states, takes actions based on a policy, and receives rewards.

### Environment

The external system that the agent interacts with. The environment presents the current state to the agent, receives the agent's actions, and generates the next state and reward according to its dynamics. Everything outside the agent's direct control is considered part of the environment.

### Reward

A scalar feedback signal indicating how well the agent is doing. The agent's objective is to maximize the expected cumulative reward over time.

### Return

The total accumulated reward from a given time step t onward. For episodic tasks, the return is the sum of rewards until the terminal state: $G_t = R_{t+1} + R_{t+2} + ... + R_T$. For continuing tasks, we use discounted return: $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$, where $\gamma$ is the discount factor $(0 \leq \gamma < 1)$.

## Multi-Armed Bandits

### k-armed bandit

A simplified RL problem with k actions (arms) where each action yields a reward from a stationary probability distribution. The goal is to maximize total reward by learning which action is best.

### Action-values

The expected reward when taking a specific action: $q(a) = \mathbb{E}[R|A=a]$

### Exploration

Trying different actions to gather information about their values.

### Exploitation

Selecting the action that currently appears best based on known values.

### Online learning

Learning from data sequentially as it becomes available, updating estimates after each observation.

### General online update equation

NewEstimate = OldEstimate + StepSize[Target - OldEstimate]

Where:

-   NewEstimate is the updated value
-   OldEstimate is the current value
-   StepSize controls how much we adjust our estimate (learning rate $\alpha$)
-   Target is the observed value we're moving toward
-   [Target - OldEstimate] represents the error in our current estimate

### Step size

The learning rate $\alpha$ that determines how much new information affects existing estimates. When constant, favors recent observations.

### Stationarity vs Non-stationarity

-   **Stationary**: The underlying reward distributions do not change over time
-   **Non-stationary**: The underlying reward distributions change over time

### Epsilon-greedy algorithm

With probability 1-ε, select the action with highest estimated value (exploit). With probability ε, select a random action (explore).

**Simple Epsilon-Greedy Bandit Algorithm**:

```
Initialize, for a = 1 to k:
    Q(a) ← 0
    N(a) ← 0
Loop forever:
    A ← argmax_a Q(a) with probability 1-ε
          a random action with probability ε (breaking ties randomly)
    R ← bandit(A)
    N(A) ← N(A) + 1
    Q(A) ← Q(A) + (1/N(A))[R - Q(A)]

```

Where:

-   Q(a) is the estimated value of action a
-   N(a) is the number of times action a was selected
-   ε is the exploration probability
-   bandit(A) returns a reward for taking action A

### Optimistic initial values

Initialize action-value estimates optimistically high to encourage exploration naturally.

### Upper Confidence Bound action selection method

Select action that maximizes: $Q(a) + c\sqrt{\frac{\ln(t)}{N(a)}}$

-   Q(a) is the estimated value of action a
-   N(a) is the number of times action a was selected
-   c controls exploration level
-   t is the time step

## Markov Decision Processes (MDPs)

### MDP

A mathematical framework for modeling decision-making where outcomes are partly random and partly under agent control. Defined by states, actions, transition probabilities, and rewards.

### Dynamics

The state transition probability function: $p(s'|s,a)$ - probability of transitioning to state $s'$ from state $s$ after taking action $a$.

### Episode, Episodic task

An episode is a complete sequence from initial to terminal state. Episodic tasks have well-defined endings.

### Discounting

Reducing the importance of future rewards by a factor $\gamma$ $(0 \leq \gamma \leq 1)$ per time step. The present value of a future reward $R$ at $t$ steps in the future is $\gamma^t \times R$.

### Continuous vs Non-continuous

-   **Continuous**: State/action spaces have infinite possible values
-   **Non-continuous (Discrete)**: State/action spaces have finite possible values

### Episodic vs Continuing

-   **Episodic**: Tasks that terminate
-   **Continuing**: Tasks that continue indefinitely

### Policy

A mapping from states to actions (or probability distributions over actions):

-   **Deterministic policy**: $\pi(s) = a$
-   **Stochastic policy**: $\pi(a|s)$ = probability of taking action $a$ in state $s$
-   **Valid policy**: For stochastic policies, $\sum_a \pi(a|s) = 1$ for all states

### State-value functions

The expected return starting from state $s$ and following policy $\pi$: $v_{\pi}(s) = \mathbb{E}_{\pi}[G_t | S_t = s]$

### Action-value functions

The expected return starting from state $s$, taking action $a$, and following policy $\pi$: $q_{\pi}(s,a) = \mathbb{E}_{\pi}[G_t | S_t = s, A_t = a]$

### Bellman equations for state-value functions

$$v_{\pi}(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)[r + \gamma v_{\pi}(s')]$$

### Bellman equations for action-value functions

$$q_{\pi}(s,a) = \sum_{s',r} p(s',r|s,a)[r + \gamma \sum_{a'} \pi(a'|s')q_{\pi}(s',a')]$$

### Optimal policy

A policy $\pi_{*}$ that achieves maximum expected return from all states. For any other policy $\pi$, $v_{\pi_{*}}(s) \geq v_{\pi}(s)$ for all states $s$.

### Bellman optimality equation for state-value functions

$$v_{*}(s) = \max_a \sum_{s',r} p(s',r|s,a)[r + \gamma v_{*}(s')]$$

### Bellman optimality equation for action-value functions

$$q_{*}(s,a) = \sum_{s',r} p(s',r|s,a)[r + \gamma \max_{a'} q_{*}(s',a')]$$

### Bellman equations vs Bellman optimality equations

-   **Bellman equations**: Express value functions in terms of expected next state values under a specific policy
-   **Bellman optimality equations**: Express optimal value functions in terms of maximum expected next state values over all possible actions

### Optimal value function verification

If a value function satisfies the Bellman optimality equation, it must be optimal. We can denote the optimal value function as $v_{*}$ and optimal action-value function as $q_{*}$.

## Dynamic Programming (DP)

### Policy evaluation (prediction)

Computing the state-value function $v_{\pi}$ for a given policy $\pi$.

### Policy interaction (control)

Finding an optimal policy.

### Dynamic programming

A collection of algorithms that compute optimal policies given a perfect model of the environment as an MDP. Limited by:

-   Requires complete, accurate model of environment
-   Requires significant computation
-   Assumes finite state and action spaces

### DP algorithm for estimating value functions under policy

Iterative application of Bellman equation as an update rule: $$v_{k+1}(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)[r + \gamma v_k(s')]$$

**Iterative Policy Evaluation Algorithm**:

```
Input: π, the policy to be evaluated
Algorithm parameter: a small threshold θ > 0 determining accuracy of estimation 
Initialize V(s), for all s ∈ S+, arbitrarily except that V(terminal) = 0

Loop:
    Δ ← 0
    Loop for each s ∈ S:
        v ← V(s)
        V(s) ← Σ_a π(a|s) Σ_{s',r} p(s',r|s,a)[r + γV(s')]
        Δ ← max(Δ, |v - V(s)|)
    until Δ < θ

```

Where:

-   V(s) is the value function for state s
-   π(a|s) is the probability of taking action a in state s under policy π
-   p(s',r|s,a) is the probability of transitioning to state s' with reward r
-   θ is the convergence threshold
-   Δ tracks the maximum change in value estimates

### Policy improvement theorem

For any policy $\pi$, if we improve the policy at a single state $s$ by choosing the greedy action, the resulting policy is at least as good as the original.

### Policy iteration for optimal policies/value functions

1.  Policy evaluation: compute $v_{\pi}$
2.  Policy improvement: make policy greedy with respect to $v_{\pi}$
3.  Repeat until policy stabilizes

**Policy Iteration Algorithm**:

```
1. Initialization
   V(s) ∈ R and π(s) ∈ A(s) arbitrarily for all s ∈ S

2. Policy Evaluation
   Loop:
       Δ ← 0
       Loop for each s ∈ S:
           v ← V(s)
           V(s) ← Σ_{s',r} p(s',r|s,π(s))[r + γV(s')]
           Δ ← max(Δ, |v - V(s)|)
       until Δ < θ (a small positive number determining the accuracy of estimation)

3. Policy Improvement
   policy-stable ← true
   For each s ∈ S:
       old-action ← π(s)
       π(s) ← argmax_a Σ_{s',r} p(s',r|s,a)[r + γV(s')]
       If old-action ≠ π(s), then policy-stable ← false

   If policy-stable, then stop and return V ≈ v_{*} and π ≈ π_{*}; else go to 2

```

Where:

-   V(s) is the value function for state s
-   π(s) is the policy (action) for state s
-   p(s',r|s,a) is the probability of transitioning to state s' with reward r
-   θ is the convergence threshold for policy evaluation
-   policy-stable indicates whether the policy has changed during improvement

### Generalized Policy Iteration (GPI)

The general idea of alternating policy evaluation and policy improvement to converge to optimal policy and value function.

### Value iteration

Combines policy evaluation and improvement into a single update: $$v_{k+1}(s) = \max_a \sum_{s',r} p(s',r|s,a)[r + \gamma v_k(s')]$$

**Value Iteration Algorithm**:

```
Algorithm parameter: a small threshold θ > 0 determining accuracy of estimation
Initialize V(s), for all s ∈ S+, arbitrarily except that V(terminal) = 0

Loop:
    Δ ← 0
    Loop for each s ∈ S:
        v ← V(s)
        V(s) ← max_a Σ_{s',r} p(s',r|s,a)[r + γV(s')]
        Δ ← max(Δ, |v - V(s)|)
    until Δ < θ

Output a deterministic policy, π ≈ π_{*}, such that:
π(s) = argmax_a Σ_{s',r} p(s',r|s,a)[r + γV(s')]

```

Where:

-   V(s) is the value function for state s
-   p(s',r|s,a) is the probability of transitioning to state s' with reward r
-   θ is the convergence threshold
-   Δ tracks the maximum change in value estimates
-   π(s) is the resulting optimal policy

### Synchronous vs Asynchronous DP methods

-   **Synchronous**: Update all states in parallel using values from previous iteration
-   **Asynchronous**: Update states in any order, using most recent values of other states

### Bootstrapping

Using estimates of value functions to update other estimates of value functions. Core idea behind DP and temporal-difference learning methods.
