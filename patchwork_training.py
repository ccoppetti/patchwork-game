import os
from pathlib import Path
import torch as th
import numpy as np
from typing import Dict, List, Optional, Tuple, Type, Union
from dataclasses import dataclass
import yaml
import json
import logging
from datetime import datetime

# Add this logging configuration block right here, before any other code
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for maximum verbosity
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training_debug.log"),  # Save logs to this file
        logging.StreamHandler(),  # Also print to console
    ],
)

# Get a logger for this module
logger = logging.getLogger(__name__)
logger.info("Starting Patchwork training initialization...")


from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
import gymnasium as gym
from gymnasium import spaces

from patchwork_gym_env import PatchworkGymEnv


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""

    total_timesteps: int = 1_000_000
    n_envs: int = 4
    batch_size: int = 64
    n_epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.02
    features_dim: int = 256
    policy_hidden_sizes: List[int] = (256, 256)
    value_hidden_sizes: List[int] = (256, 256)
    save_freq: int = 10000
    eval_freq: int = 10000
    n_eval_episodes: int = 20
    log_interval: int = 1

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def save(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f)


class PatchworkFeatureExtractor(BaseFeaturesExtractor):
    """Custom feature extractor for the Patchwork game state."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 256,
        activation_fn: Type[th.nn.Module] = th.nn.ReLU,
    ):
        """Initialize the feature extractor."""
        super().__init__(observation_space, features_dim)

        # Board processing network
        self.board_net = th.nn.Sequential(
            th.nn.Conv2d(2, 32, kernel_size=3, padding=1),
            activation_fn(),
            th.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            activation_fn(),
            th.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            activation_fn(),
            th.nn.AdaptiveAvgPool2d((3, 3)),  # Reduce spatial dimensions
            th.nn.Flatten(),
            th.nn.Linear(64 * 3 * 3, 256),
            activation_fn(),
            th.nn.Dropout(0.1),
        )

        # Patch processing network
        self.patch_net = th.nn.Sequential(
            th.nn.Conv2d(1, 16, kernel_size=3, padding=1),
            activation_fn(),
            th.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            activation_fn(),
            th.nn.AdaptiveAvgPool2d((2, 2)),
            th.nn.Flatten(),
            th.nn.Linear(32 * 2 * 2, 64),
            activation_fn(),
            th.nn.Dropout(0.1),
        )

        # Network for patch properties
        self.property_net = th.nn.Sequential(
            th.nn.Linear(3, 32),
            activation_fn(),
            th.nn.Linear(32, 32),
            activation_fn(),
            th.nn.Dropout(0.1),
        )

        # Game state processing network
        self.state_net = th.nn.Sequential(
            th.nn.Linear(5, 32),
            activation_fn(),
            th.nn.Linear(32, 32),
            activation_fn(),
            th.nn.Dropout(0.1),
        )

        # Final combination network with layer normalization
        self.combine_net = th.nn.Sequential(
            th.nn.Linear(256 + 3 * (64 + 32) + 32, features_dim),
            th.nn.LayerNorm(features_dim),
            activation_fn(),
            th.nn.Dropout(0.1),
        )

    def forward(self, observations: Dict[str, th.Tensor]) -> th.Tensor:
        """Process the observations through the networks."""
        # Process boards
        boards = th.stack(
            [observations["player1_board"], observations["player2_board"]], dim=1
        ).float()
        board_features = self.board_net(boards)

        # Process patches
        patch_features_list = []
        for i in range(3):  # Process each patch
            shape = observations["patch_shapes"][:, i].unsqueeze(1).float()
            mask = observations["patch_masks"][:, i].unsqueeze(1).float()
            shape_features = self.patch_net(shape) * mask

            properties = th.stack(
                [
                    observations["patch_buttons"][:, i],
                    observations["patch_times"][:, i],
                    observations["patch_incomes"][:, i],
                ],
                dim=1,
            ).float()
            property_features = self.property_net(properties) * mask.squeeze(1)

            patch_features_list.extend([shape_features, property_features])

        # Process game state
        state = th.stack(
            [
                observations["player1_buttons"].squeeze(-1),
                observations["player2_buttons"].squeeze(-1),
                observations["player1_position"].squeeze(-1),
                observations["player2_position"].squeeze(-1),
                observations["current_player"].float(),
            ],
            dim=1,
        ).float()
        state_features = self.state_net(state)

        # Combine all features
        combined = th.cat(
            [board_features] + patch_features_list + [state_features], dim=1
        )
        return self.combine_net(combined)


class TrainingMetricsCallback(BaseCallback):
    """Callback for collecting training metrics."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.metrics = {
            "rewards": [],
            "episode_lengths": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy_loss": [],
        }

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            self.metrics["rewards"].append(
                np.mean([ep["r"] for ep in self.model.ep_info_buffer])
            )
            self.metrics["episode_lengths"].append(
                np.mean([ep["l"] for ep in self.model.ep_info_buffer])
            )
        return True

    def save_metrics(self, path: str) -> None:
        """Save collected metrics to JSON file."""
        with open(path, "w") as f:
            json.dump(self.metrics, f)


class PatchworkTrainer:
    """Handles training of the Patchwork AI agent."""

    def __init__(
        self,
        config: TrainingConfig,
        model_dir: str = "patchwork_model",
        device: str = "auto",
    ):
        """Initialize the trainer."""
        self.config = config
        self.model_dir = Path(model_dir)
        self.device = self._get_device(device)

        # Create necessary directories
        self._setup_directories()

        # Save configuration
        self.config.save(self.model_dir / "config.yaml")

        logger.info(f"Initialized trainer with device: {self.device}")
        logger.info(f"Model directory: {self.model_dir}")

    def _get_device(self, device: str) -> str:
        """Determine the appropriate device for training."""
        if device == "auto":
            return "cuda" if th.cuda.is_available() else "cpu"
        return device

    def _setup_directories(self) -> None:
        """Create necessary directories for saving models and logs."""
        directories = [
            self.model_dir,
            self.model_dir / "checkpoints",
            self.model_dir / "logs",
            self.model_dir / "eval_logs",
            self.model_dir / "best_model",
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def _make_env(self, rank: int, seed: int = 0, eval_env: bool = False) -> callable:
        """Create a wrapped, monitored Patchwork environment."""

        def _init() -> Monitor:
            try:
                env = PatchworkGymEnv()
                env.reset(seed=(seed + rank))

                # Wrap environment with monitor
                log_dir = self.model_dir / ("eval_logs" if eval_env else "logs")
                env = Monitor(
                    env,
                    str(log_dir / str(rank)),
                    info_keywords=("player1_score", "player2_score"),
                )
                return env

            except Exception as e:
                logger.error(
                    f"Environment initialization failed for rank {rank}: {str(e)}"
                )
                raise RuntimeError(
                    f"Environment initialization failed: {str(e)}"
                ) from e

        return _init

    def _setup_callbacks(self) -> CallbackList:
        """Set up training callbacks."""
        # Checkpoint saving
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.save_freq,
            save_path=str(self.model_dir / "checkpoints"),
            name_prefix="patchwork_model",
        )

        # Evaluation
        eval_env = SubprocVecEnv([self._make_env(i, eval_env=True) for i in range(2)])
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(self.model_dir / "best_model"),
            log_path=str(self.model_dir / "eval_logs"),
            eval_freq=self.config.eval_freq,
            n_eval_episodes=self.config.n_eval_episodes,
            deterministic=True,
            render=False,
        )

        # Metrics collection
        metrics_callback = TrainingMetricsCallback()

        return CallbackList([checkpoint_callback, eval_callback, metrics_callback])

    def train(self) -> PPO:
        """Train the agent.

        Returns:
            PPO: Trained model
        """
        try:
            logger.info("Creating training environment...")
            # Create vectorized environment
            env = SubprocVecEnv([self._make_env(i) for i in range(self.config.n_envs)])
            env = VecMonitor(env)

            logger.info("Initializing PPO model...")
            # Initialize model
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=self.config.learning_rate,
                n_steps=2048,
                batch_size=self.config.batch_size,
                n_epochs=self.config.n_epochs,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                clip_range=self.config.clip_range,
                clip_range_vf=None,
                normalize_advantage=True,
                ent_coef=self.config.ent_coef,
                vf_coef=self.config.vf_coef,
                max_grad_norm=self.config.max_grad_norm,
                use_sde=False,
                sde_sample_freq=-1,
                target_kl=self.config.target_kl,
                tensorboard_log=str(self.model_dir / "tb_logs"),
                policy_kwargs=dict(
                    features_extractor_class=PatchworkFeatureExtractor,
                    features_extractor_kwargs=dict(
                        features_dim=self.config.features_dim
                    ),
                    net_arch=dict(
                        pi=list(self.config.policy_hidden_sizes),
                        vf=list(self.config.value_hidden_sizes),
                    ),
                ),
                verbose=1,
                device=self.device,
            )

            # Set up callbacks
            logger.info("Setting up training callbacks...")
            callbacks = self._setup_callbacks()

            # Train model
            logger.info("Starting training...")
            model.learn(
                total_timesteps=self.config.total_timesteps,
                callback=callbacks,
                log_interval=self.config.log_interval,
            )

            # Save final model
            final_model_path = self.model_dir / "final_model.zip"
            model.save(str(final_model_path))
            logger.info(f"Final model saved to {final_model_path}")

            # Save training metrics
            metrics_callback = callbacks.callbacks[2]  # TrainingMetricsCallback
            metrics_callback.save_metrics(self.model_dir / "training_metrics.json")

            return model

        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise


def _make_env(self, rank: int, seed: int = 0, eval_env: bool = False) -> callable:
    """Create a wrapped, monitored Patchwork environment."""

    def _init() -> Monitor:
        try:
            env = PatchworkGymEnv()
            # Test initialization
            observation, _ = env.reset(seed=(seed + rank))
            if observation is None:
                raise RuntimeError(
                    "Environment initialization failed - no initial observation"
                )

            env.reset(seed=(seed + rank))

            # Wrap environment with monitor
            log_dir = self.model_dir / ("eval_logs" if eval_env else "logs")
            env = Monitor(
                env,
                str(log_dir / str(rank)),
                info_keywords=("player1_score", "player2_score"),
            )
            return env

        except Exception as e:
            logger.error(f"Environment initialization failed for rank {rank}: {str(e)}")
            raise RuntimeError(f"Environment initialization failed: {str(e)}") from e

    return _init

    def _setup_callbacks(self) -> CallbackList:
        """Set up training callbacks."""
        # Checkpoint saving
        checkpoint_callback = CheckpointCallback(
            save_freq=self.config.save_freq,
            save_path=str(self.model_dir / "checkpoints"),
            name_prefix="patchwork_model",
        )

        # Evaluation
        eval_env = SubprocVecEnv([self._make_env(i, eval_env=True) for i in range(2)])
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(self.model_dir / "best_model"),
            log_path=str(self.model_dir / "eval_logs"),
            eval_freq=self.config.eval_freq,
            n_eval_episodes=self.config.n_eval_episodes,
            deterministic=True,
            render=False,
        )

        # Metrics collection
        metrics_callback = TrainingMetricsCallback()

        return CallbackList([checkpoint_callback, eval_callback, metrics_callback])

    def train(self) -> PPO:
        """Train the agent."""
        try:
            # Create vectorized environment
            env = SubprocVecEnv([self._make_env(i) for i in range(self.config.n_envs)])
            env = VecMonitor(env)

            # Initialize model
            model = PPO(
                "MlpPolicy",  # Using default policy for now
                env,
                learning_rate=self.config.learning_rate,
                n_steps=2048,
                batch_size=self.config.batch_size,
                n_epochs=self.config.n_epochs,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                clip_range=self.config.clip_range,
                clip_range_vf=None,
                normalize_advantage=True,
                ent_coef=self.config.ent_coef,
                vf_coef=self.config.vf_coef,
                max_grad_norm=self.config.max_grad_norm,
                use_sde=False,
                sde_sample_freq=-1,
                target_kl=self.config.target_kl,
                tensorboard_log=str(self.model_dir / "tb_logs"),
                policy_kwargs=dict(
                    features_extractor_class=PatchworkFeatureExtractor,
                    features_extractor_kwargs=dict(
                        features_dim=self.config.features_dim
                    ),
                    net_arch=dict(
                        pi=list(self.config.policy_hidden_sizes),
                        vf=list(self.config.value_hidden_sizes),
                    ),
                ),
                verbose=1,
                device=self.device,
            )

            # Set up callbacks
            callbacks = self._setup_callbacks()

            # Train model
            logger.info("Starting training...")
            model.learn(
                total_timesteps=self.config.total_timesteps,
                callback=callbacks,
                log_interval=self.config.log_interval,
            )

            # Save final model
            final_model_path = self.model_dir / "final_model.zip"
            model.save(str(final_model_path))
            logger.info(f"Final model saved to {final_model_path}")

            # Save training metrics
            metrics_callback = callbacks.callbacks[2]  # TrainingMetricsCallback
            metrics_callback.save_metrics(self.model_dir / "training_metrics.json")

            return model

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise


def evaluate_model(
    model: PPO, env: PatchworkGymEnv, n_eval_episodes: int = 100
) -> Tuple[float, float]:
    """Evaluate a trained model."""
    episode_rewards = []

    for episode in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        episode_rewards.append(episode_reward)

    return np.mean(episode_rewards), np.std(episode_rewards)


def main():
    """Main training script."""
    try:
        # Load configuration
        config_path = "training_config.yaml"
        if Path(config_path).exists():
            config = TrainingConfig.from_yaml(config_path)
        else:
            config = TrainingConfig()
            config.save(config_path)

        # Initialize trainer
        trainer = PatchworkTrainer(
            config=config, model_dir="patchwork_model", device="auto"
        )

        # Train model
        model = trainer.train()

        # Evaluate final model
        logger.info("Evaluating final model...")
        eval_env = PatchworkGymEnv()
        mean_reward, std_reward = evaluate_model(model, eval_env, n_eval_episodes=100)
        logger.info(f"Final evaluation: {mean_reward:.2f} ± {std_reward:.2f}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
