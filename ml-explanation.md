# Modern Machine Learning Practices: A Guide Through the Patchwork AI Implementation

## Introduction

This document explains the implementation of an AI for the Patchwork board game, using it as a lens to understand modern machine learning practices. If you've been away from ML for the past decade, you'll find that while the fundamental principles remain the same, the tools, techniques, and best practices have evolved significantly.

## Key Modern ML Concepts

### 1. Deep Reinforcement Learning (Deep RL)

#### Then vs Now
- **Then**: RL typically used tabular methods or simple function approximators
- **Now**: Deep neural networks are the standard function approximators in RL, handling complex state spaces effectively

#### Why This Matters for Patchwork
Patchwork has a complex state space with:
- Two 9x9 boards
- Multiple patches with various shapes and rotations
- Multiple resource types (buttons, time)
- Sequential decision-making

Traditional methods would struggle with this complexity, but deep RL handles it naturally.

### 2. The Rise of PyTorch

#### Then vs Now
- **Then**: Frameworks like Theano, early TensorFlow with static graphs
- **Now**: PyTorch dominates research due to:
  - Dynamic computation graphs
  - Pythonic programming model
  - Extensive ecosystem

#### In Our Implementation
We use PyTorch for:
```python
class PatchworkFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        # Modern PyTorch patterns like nn.Sequential
        self.board_net = th.nn.Sequential(
            th.nn.Conv2d(2, 32, kernel_size=3, padding=1),
            th.nn.ReLU(),
            # ...
        )
```

This code is more readable and maintainable than older frameworks.

### 3. Modern RL Algorithms

#### Then vs Now
- **Then**: Q-learning, SARSA, basic policy gradient methods
- **Now**: Advanced policy gradient methods like PPO (Proximal Policy Optimization)

#### Why We Use PPO
PPO (2017) has become the go-to algorithm because it:
- Is more stable than older methods
- Requires less hyperparameter tuning
- Prevents catastrophic policy updates
- Works well across many domains

Our implementation uses PPO through stable-baselines3:
```python
model = PPO(
    PatchworkPolicy,
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    # ...
)
```

### 4. Environment Standardization

#### Then vs Now
- **Then**: Custom environment implementations
- **Now**: Standardized environments through OpenAI Gym (now Gymnasium)

#### Our Environment Implementation
```python
class PatchworkGymEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Dict({
            'player1_board': gym.spaces.Box(
                low=0, high=1, shape=(9, 9), dtype=np.int8
            ),
            # ...
        })
        self.action_space = gym.spaces.MultiDiscrete([4, 4, 9, 9])
```

Benefits:
- Compatibility with existing RL libraries
- Standard interface for environments
- Easy benchmarking and comparison

### 5. Neural Network Architecture Design

#### Then vs Now
- **Then**: Simple feedforward networks, basic CNNs
- **Now**: Specialized architectures for different input types

#### Our Feature Extractor
We use different network components for different aspects of the game state:

1. **Board State Processing**:
```python
self.board_net = th.nn.Sequential(
    th.nn.Conv2d(2, 32, kernel_size=3, padding=1),
    activation_fn(),
    th.nn.Conv2d(32, 64, kernel_size=3, padding=1),
    # ...
)
```
- Uses CNNs to capture spatial patterns
- Maintains board structure information

2. **Patch Processing**:
```python
self.patch_net = th.nn.Sequential(
    th.nn.Conv2d(1, 16, kernel_size=3, padding=1),
    # ...
)
```
- Processes patch shapes independently
- Captures patch geometry features

3. **Game State Processing**:
```python
self.state_net = th.nn.Sequential(
    th.nn.Linear(5, 32),  # buttons, positions, current_player
    # ...
)
```
- Handles scalar game state variables

### 6. Training Infrastructure

#### Then vs Now
- **Then**: Single-process training, basic logging
- **Now**: Parallel training, comprehensive monitoring

#### Modern Training Features in Our Implementation

1. **Parallel Environment Processing**:
```python
env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
```
- Runs multiple environments in parallel
- Increases training throughput

2. **Comprehensive Callbacks**:
```python
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=save_dir,
    name_prefix="patchwork_model"
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=f"{save_dir}/best_model",
    # ...
)
```
- Regular model checkpointing
- Automated evaluation
- Performance tracking

3. **Tensorboard Integration**:
```python
model = PPO(
    # ...
    tensorboard_log=f"{save_dir}/tb_logs",
    # ...
)
```
- Real-time training visualization
- Performance metrics tracking
- Learning curve analysis

## Implementation Deep Dive

### 1. State Representation

The state space is represented as a dictionary with:
- Board states (9x9 binary matrices)
- Available patches (shapes, costs, rotations)
- Game state variables (buttons, positions)

This structured representation allows the neural network to process different aspects of the game state appropriately.

### 2. Action Space

Actions are represented as a MultiDiscrete space:
```python
self.action_space = gym.spaces.MultiDiscrete([
    4,    # patch_idx (-1 to 2)
    4,    # rotation (0-3)
    9,    # row (0-8)
    9     # col (0-8)
])
```

This design:
- Handles invalid actions through the environment
- Maintains action independence
- Supports both patch placement and passing

### 3. Reward Design

The reward function focuses on score difference:
```python
reward = (current_scores[0] - prev_scores[0]) - (current_scores[1] - prev_scores[1])
```

This approach:
- Provides immediate feedback
- Aligns with game objectives
- Handles both short-term and long-term strategy

## Best Practices and Tips

1. **Environment Design**:
   - Make invalid actions return to the same state
   - Use normalized observation values
   - Include relevant information in info dict

2. **Network Architecture**:
   - Process different input types separately
   - Use appropriate layer types for each input
   - Combine features meaningfully

3. **Training**:
   - Start with small networks and scale up
   - Monitor training progress with Tensorboard
   - Save models regularly

4. **Evaluation**:
   - Test against different opponents
   - Track multiple metrics
   - Use deterministic evaluation

## Conclusion

Modern ML, particularly in the context of deep RL, provides powerful tools for creating game-playing agents. The combination of:
- Standardized environments (Gymnasium)
- Efficient deep learning frameworks (PyTorch)
- Advanced algorithms (PPO)
- Modern training infrastructure

Makes it possible to create sophisticated agents for complex games like Patchwork. The key is understanding how to leverage these tools effectively while maintaining good software engineering practices.
