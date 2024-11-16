# Training Guide for Patchwork AI

## Setup

1. First, install the required packages:
```bash
pip install torch stable-baselines3 tensorboard gymnasium numpy
```

2. Create a new directory for your training:
```bash
mkdir patchwork_training
cd patchwork_training
```

3. Make sure you have all the necessary files in your directory:
- patchwork_gym_env.py
- patchwork_training.py
- patch_config.py
- patch_dataclass.py
- patchwork_env.py

## Basic Training

The simplest way to start training is to run the training script:

```bash
python patchwork_training.py
```

This will:
- Create a "patchwork_model" directory
- Start training with default parameters
- Save checkpoints and logs
- Run evaluation episodes periodically

## Monitor Training Progress

1. Start Tensorboard:
```bash
tensorboard --logdir patchwork_model/tb_logs
```

2. Open your browser and go to:
```
http://localhost:6006
```

You'll see real-time graphs of:
- Episode rewards
- Policy loss
- Value loss
- Learning rate
- Other metrics

## Training Options

You can customize training by modifying the parameters in `patchwork_training.py`:

```python
# Adjust these parameters at the bottom of the file
TOTAL_TIMESTEPS = 1000000  # Total training steps
N_ENVS = 8                 # Number of parallel environments
SAVE_DIR = "patchwork_model"
DEVICE = "cuda" if th.cuda.is_available() else "cpu"
```

Common adjustments:

1. **Training Length**:
```python
TOTAL_TIMESTEPS = 2000000  # Train for longer
```

2. **Parallel Environments**:
```python
N_ENVS = 4  # Reduce if you have less CPU cores
```

3. **Batch Size**:
```python
model = PPO(
    PatchworkPolicy,
    env,
    batch_size=32,  # Decrease if you have memory issues
    # ...
)
```

## Advanced Training Configuration

You can create a training configuration file for more control:

```python
# training_config.py
training_config = {
    # PPO parameters
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    
    # Training setup
    "total_timesteps": 1000000,
    "n_envs": 8,
    
    # Network architecture
    "policy_kwargs": {
        "features_dim": 256,
        "net_arch": {
            "pi": [256, 256],
            "vf": [256, 256]
        }
    },
    
    # Saving and logging
    "save_freq": 10000,
    "eval_freq": 10000,
    "n_eval_episodes": 50,
}
```

Then modify the training script to use this config:

```python
def train_with_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    model = train_patchwork_agent(**config)
    return model
```

## Training Stages

For better results, consider training in stages:

1. **Initial Training**:
```python
# Start with simpler network and higher learning rate
initial_config = {
    "learning_rate": 1e-3,
    "policy_kwargs": {
        "features_dim": 128,
        "net_arch": {
            "pi": [128],
            "vf": [128]
        }
    },
    "total_timesteps": 500000,
}
```

2. **Fine-tuning**:
```python
# Load initial model and continue training with more complex settings
model = PPO.load("patchwork_model/best_model")
model.learn(
    total_timesteps=500000,
    learning_rate=1e-4,
    # ...
)
```

## Evaluating Training Progress

1. Regular evaluation:
```python
# Run evaluation every 50000 steps
eval_callback = EvalCallback(
    eval_env,
    eval_freq=50000,
    n_eval_episodes=100,
    deterministic=True
)
```

2. Test against different opponents:
```python
def evaluate_against_opponent(model_path, opponent_type="random"):
    model = PPO.load(model_path)
    
    # Create evaluation environment with specific opponent
    eval_env = PatchworkGymEnv(opponent=opponent_type)
    
    # Run evaluation episodes
    results = evaluate_agent(model, eval_env, n_eval_episodes=100)
    return results
```

## Common Issues and Solutions

1. **Out of Memory**:
- Reduce batch_size
- Reduce n_steps
- Reduce number of parallel environments
- Use smaller network architecture

2. **Slow Training**:
- Increase number of parallel environments
- Use GPU if available
- Reduce evaluation frequency
- Simplify network architecture initially

3. **Poor Performance**:
- Train for more steps
- Adjust learning rate
- Modify reward function
- Add curriculum learning

## Best Practices

1. **Save Frequently**:
```python
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="patchwork_model",
    name_prefix="patchwork"
)
```

2. **Track Multiple Metrics**:
```python
# In your evaluation callback
info_keywords = (
    "player1_score",
    "player2_score",
    "game_length",
    "invalid_moves"
)
```

3. **Use Git for Version Control**:
- Track your code changes
- Save different configurations
- Document what works and what doesn't

4. **Document Your Training**:
```python
# Save training parameters with the model
model_info = {
    "training_steps": TOTAL_TIMESTEPS,
    "architecture": model.policy_kwargs,
    "final_reward": mean_reward,
    "win_rate": win_rate,
    "date": datetime.now().strftime("%Y-%m-%d_%H-%M")
}

with open(f"{SAVE_DIR}/model_info.json", "w") as f:
    json.dump(model_info, f, indent=4)
```

## Next Steps

After training:

1. Analyze performance with:
```python
evaluate_agent("patchwork_model/best_model", n_eval_episodes=1000)
```

2. Use the model in your GUI:
```python
model = PPO.load("patchwork_model/best_model")
action, _ = model.predict(observation, deterministic=True)
```

3. Continue training from checkpoints if needed:
```python
model = PPO.load("patchwork_model/checkpoint_100000")
model.learn(total_timesteps=100000)
```
