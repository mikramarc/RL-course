# Function Approximation in Reinforcement Learning

## On-Policy Prediction with Approximation

### Function Approximation
Using parameterized functions to approximate value functions instead of storing a separate value for each state. Enables generalization to unseen states and handling of large/continuous state spaces.

### Update Target
The ideal value toward which we update our approximation. In TD methods, the update target is $R + \gamma \hat{v}(S', \mathbf{w})$, while in Monte Carlo methods, the target is the return $G_t$.

### Supervised Learning Connection
Function approximation in RL is similar to supervised learning, with state-value pairs as training examples, but differs because targets are generated from the agent's own estimates.

### Prediction Objective
Minimizing the mean squared value error (MSVE) over the on-policy distribution:

$\overline{VE}(\mathbf{w}) = \sum_{s \in \mathcal{S}} \mu(s) [v_{\pi}(s) - \hat{v}(s, \mathbf{w})]^2$

where $\mu(s)$ is the on-policy distribution.

### Mean Squared Value Error (MSVE)
The average squared difference between the true value function $v_{\pi}(s)$ and its approximation $\hat{v}(s, \mathbf{w})$, weighted by the on-policy distribution:

$\overline{VE}(\mathbf{w}) = \mathbb{E}_{\mu}[(v_{\pi}(s) - \hat{v}(s, \mathbf{w}))^2]$

### On-Policy Distribution
The distribution of states encountered when following policy $\pi$. For continuing tasks, it's the stationary distribution under $\pi$. For episodic tasks, it depends on the starting state distribution and transition dynamics under $\pi$.

### Global/Local Optimum
- **Global optimum**: The parameter setting that gives the lowest possible error over the entire parameter space
- **Local optimum**: A parameter setting where small changes in any direction would increase the error, but may not be the global best

### Stochastic Gradient
Optimization approach that updates parameters using estimates of the gradient based on samples, rather than computing the full gradient:
$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \nabla \hat{J}(\mathbf{w}_t)$
where $\nabla \hat{J}(\mathbf{w}_t)$ is a sample-based estimate of the gradient.

### Semi-Gradient
A partial gradient update that treats some estimated quantities as constants. In TD learning, we calculate the gradient of the TD target as if the target were a constant, ignoring that it depends on the weights.

### Unbiased Estimate
An estimate whose expected value equals the true value. Monte Carlo returns are unbiased estimates of the true value function, while TD targets are biased due to bootstrapping.

### Gradient Monte Carlo Algorithm
```
Initialize weight vector w arbitrarily
Repeat for each episode:
    Generate an episode S₀, A₀, R₁, S₁, A₁, ..., Sₜ following π
    For each step t = 0, 1, ..., T-1:
        G ← return from step t
        w ← w + α[G - v̂(Sₜ,w)]∇v̂(Sₜ,w)
```

### Semi-Gradient TD(0)
```
Initialize weight vector w arbitrarily
Repeat for each episode:
    Initialize S
    For each step:
        Take action A according to π
        Observe R, S'
        w ← w + α[R + γv̂(S',w) - v̂(S,w)]∇v̂(S,w)
        S ← S'
    Until S is terminal
```

### TD Fixed Point
The solution where the expected TD update is zero. Semi-gradient TD(0) converges to this fixed point:
$\mathbf{w_{TD}} = \mathbf{w}$ such that $\mathbb{E}[(R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}) - \hat{v}(S_t, \mathbf{w}))\nabla \hat{v}(S_t, \mathbf{w})] = \mathbf{0}$

### n-Step Semi-Gradient TD
Uses an n-step return as the update target:

$G_{t:t+n} = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{n-1}R_{t+n} + \gamma^n \hat{v}(S_{t+n}, \mathbf{w})$

$\mathbf{w}_{t+n} = \mathbf{w}_{t+n-1} + \alpha [G_{t:t+n} - \hat{v}(S_t, \mathbf{w}_{t+n-1})]\nabla \hat{v}(S_t, \mathbf{w}_{t+n-1})$


## Constructing Features for Prediction

### Generalization
The ability to make accurate predictions for states that have never been experienced, based on similarities to observed states.

### Discrimination
The ability to distinguish between states that require different value estimates and treat them differently.

### Feature Vector
A vector representation of a state $\mathbf{x}(s) = (x_1(s), x_2(s), ..., x_d(s))^T$ where each element $x_i(s)$ is a feature of the state.

### Feature
A specific aspect or property extracted from a state that is used as input to a function approximator. Good features capture relevant information for prediction.

### Basis Function
A function that maps a state to a feature value. A set of basis functions forms the feature representation of a state.

### Linear Approximation
A function approximation where the output is a linear combination of features:
$\hat{v}(s, \mathbf{w}) = \mathbf{w}^T \mathbf{x}(s) = \sum_{i=1}^d w_i x_i(s)$
where $\mathbf{w}$ is the weight vector and $\mathbf{x}(s)$ is the feature vector for state $s$.

### State Aggregation
A simple form of function approximation where states are grouped into clusters, and all states in a cluster share the same value. Each state belongs to exactly one group and is represented by a one-hot encoded feature vector.

### Coarse Coding
A representation method using multiple overlapping receptive fields (features), where each state activates multiple features. More overlapping leads to better generalization but less discrimination.

### Tile Coding
A specific form of coarse coding using multiple overlapping tilings, where each tiling divides the state space into non-overlapping tiles. Different tilings are offset from each other to create multiple levels of generalization.

### Hashing
A technique used with tile coding to reduce memory requirements by mapping multiple tile indices to a smaller set of features, allowing a compact representation with controlled collisions.

### Radial Basis Functions (RBF)
Features that activate based on the distance from feature centers. The activation decreases with distance according to a Gaussian or other radial function:
$\phi_i(s) = \exp(-\frac{||s - c_i||^2}{2\sigma_i^2})$
where $c_i$ is the center and $\sigma_i$ controls the width of the RBF.

### Artificial Neural Networks (ANN)
Function approximators composed of layers of interconnected units (neurons) that transform inputs through nonlinear activation functions. Used for complex, hierarchical feature extraction.

### Backpropagation
Algorithm for training neural networks by computing gradients of the error with respect to the weights, using the chain rule to propagate gradients backward through the network.

### Overfitting
When a function approximator models the training data too closely, capturing noise rather than the underlying pattern, leading to poor generalization to new data.

### Cross Validation
Technique to assess model performance by partitioning data into training and validation sets, allowing evaluation on data not used during training to detect overfitting.

### Regularization
Methods to prevent overfitting by adding constraints or penalties to the learning process, such as L1/L2 regularization which penalizes large weights.

### Batch Normalization
Technique in deep learning that normalizes the inputs to each layer, reducing internal covariate shift and enabling faster training with higher learning rates.

### Deep Residual Learning
Architecture using residual connections (shortcuts) between layers that allow gradients to flow more effectively through deep networks, enabling training of very deep networks.

## Control with Approximation

### Episodic Semi-Gradient Control
Control method using function approximation in episodic tasks, applying semi-gradient updates to action-value functions.

### Episodic Semi-Gradient Sarsa Algorithm
```
Initialize weight vector w arbitrarily
Repeat for each episode:
    Initialize S, A (e.g., ε-greedy based on q̂(S,·,w))
    For each step:
        Take action A, observe R, S'
        If S' is terminal:
            w ← w + α[R - q̂(S,A,w)]∇q̂(S,A,w)
            Go to next episode
        Choose A' as a function of q̂(S',·,w) (e.g., ε-greedy)
        w ← w + α[R + γq̂(S',A',w) - q̂(S,A,w)]∇q̂(S,A,w)
        S ← S'; A ← A'
```

### Average Reward
Alternative to discounted or episodic returns for continuing tasks, defined as the long-term average reward per time step:
$r(\pi) = \lim_{h→∞} \frac{1}{h}\sum_{t=1}^{h} \mathbb{E}[R_t | A_t \sim \pi]$

### Settings: Episodic vs Discounted vs Average Reward
- **Episodic**: Tasks with clear endpoints, optimizes total return per episode
- **Discounted**: Tasks where future rewards are worth less than immediate rewards
- **Average Reward**: Continuing tasks where long-term rate of reward is optimized

### Ergodicity
Property of an MDP where the limiting distribution of states is independent of the starting state. In ergodic MDPs, the average reward is well-defined and independent of the starting state.

### Differential Return
The excess reward over the average reward, summed over time:
$G_t = (R_{t+1} - r(\pi)) + (R_{t+2} - r(\pi)) + ...$
where $r(\pi)$ is the average reward.

### Differential Value Function
The expected differential return starting from state $s$:
$v_{\pi}(s) = \mathbb{E}_{\pi}[G_t | S_t = s]$

### Differential Bellman Equations
The Bellman equations adjusted for average reward setting:
$v_{\pi}(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)[r - r(\pi) + v_{\pi}(s')]$

### Differential Form of TD Errors
The TD error adjusted for average reward:
$\delta_t = R_{t+1} - \bar{R}_t + \hat{v}(S_{t+1}, \mathbf{w}) - \hat{v}(S_t, \mathbf{w})$
where $\bar{R}_t$ is the current estimate of the average reward.

### Differential Semi-Gradient Sarsa
Algorithm for average reward setting:
```
Initialize weight vector w and average reward estimate R̄
Repeat for each step:
    Take action A, observe R, S'
    Choose A' using policy derived from q̂ (e.g., ε-greedy)
    δ ← R - R̄ + q̂(S',A',w) - q̂(S,A,w)
    R̄ ← R̄ + βδ                         // Update average reward estimate
    w ← w + αδ∇q̂(S,A,w)                 // Update weights
    S ← S'; A ← A'
```

### Visitation Count
The number of times a state is visited during policy execution, related to the on-policy distribution. Used in importance sampling and some exploration techniques.

## Policy Gradient

### Parametrized Policy
A policy directly represented as a parameterized function:
$\pi(a|s, \boldsymbol{\theta}) = \text{Pr}(A_t=a | S_t=s, \boldsymbol{\theta})$
where $\boldsymbol{\theta}$ is the policy parameter vector.

### Performance Maximization
Objective of policy gradient methods: find policy parameters that maximize a performance measure $J(\boldsymbol{\theta})$ such as expected return.

### Policy Gradient Method
Approach that optimizes policy parameters by following the gradient of a performance measure:
$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \alpha \nabla J(\boldsymbol{\theta}_t)$

### Actor-Critic Methods
Methods that combine policy gradient (actor) with value function approximation (critic), where the critic evaluates the actor's policy and helps reduce the variance of policy updates.

### Action Preferences
Unconstrained numerical values $h(s,a,\boldsymbol{\theta})$ representing relative preference for each action. These are converted to a probability distribution using soft-max.

### Soft-Max in Action Preferences
Converts action preferences to a probability distribution:
$\pi(a|s, \boldsymbol{\theta}) = \frac{e^{h(s,a,\boldsymbol{\theta})}}{\sum_b e^{h(s,b,\boldsymbol{\theta})}}$

### Advantages of Policy Parametrization vs Action-Value Parametrization
- Simpler to represent stochastic policies
- Better convergence properties in some cases
- Can learn policies in continuous action spaces without discretization
- Focus computation on relevant regions of action space
- Smoother policy improvement compared to greedy methods

### Performance Measure
Objective function $J(\boldsymbol{\theta})$ to be maximized in policy gradient. Common measures include:
- Expected start value: $J(\boldsymbol{\theta}) = v_{\pi_{\boldsymbol{\theta}}}(s_0)$
- Average reward: $J(\boldsymbol{\theta}) = \sum_s d_{\pi_{\boldsymbol{\theta}}}(s) \sum_a \pi_{\boldsymbol{\theta}}(a|s) \sum_{s',r} p(s',r|s,a)r$

### Policy Gradient Theorem
Key theorem that provides an analytical expression for the gradient of performance with respect to policy parameters:
$\nabla J(\boldsymbol{\theta}) \propto \sum_s \mu(s) \sum_a q_{\pi}(s,a) \nabla \pi(a|s, \boldsymbol{\theta})$
Where $\mu(s)$ is the on-policy distribution under $\pi_{\boldsymbol{\theta}}$.

### Monte Carlo Policy Gradient
Policy gradient method using Monte Carlo returns to estimate the action-value function.

### REINFORCE Algorithm
Monte Carlo policy gradient method:
```
Initialize policy parameter θ
For each episode:
    Generate an episode S₀, A₀, R₁, S₁, A₁, ..., Sₜ following π(·|·,θ)
    For each step t = 0, 1, ..., T-1:
        G ← return from step t
        θ ← θ + αGt∇ln π(At|St,θ)
```

### Eligibility Vector
The gradient of the logarithm of the policy with respect to the parameters:
$\nabla \ln \pi(a|s, \boldsymbol{\theta})$
This vector indicates how much each parameter affects the probability of taking action $a$ in state $s$.

### Baseline
A function $b(s)$ subtracted from the return to reduce variance in policy gradient methods without changing the expected gradient:
$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \alpha (G_t - b(S_t)) \nabla \ln \pi(A_t | S_t, \boldsymbol{\theta}_t)$

### REINFORCE with Baseline
REINFORCE algorithm with a state-value function as baseline:
```
Initialize policy parameter θ and state-value weights w
For each episode:
    Generate an episode S₀, A₀, R₁, S₁, A₁, ..., Sₜ following π(·|·,θ)
    For each step t = 0, 1, ..., T-1:
        G ← return from step t
        δ ← G - v̂(St,w)                 // Advantage estimate
        w ← w + αwδ∇v̂(St,w)             // Update critic
        θ ← θ + αθδ∇ln π(At|St,θ)        // Update actor
```
