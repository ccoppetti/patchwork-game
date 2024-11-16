from typing import Dict, Tuple, Any, Optional, List, Union
import gymnasium as gym
import numpy as np
from patchwork_env import PatchworkState, Move, InvalidMoveError
from patch_dataclass import Patch
import logging
from dataclasses import dataclass
from enum import IntEnum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Enum for action types
class PatchworkAction(IntEnum):
    """Enumeration of possible action types."""

    PASS = -1
    PLACE_PATCH_0 = 0
    PLACE_PATCH_1 = 1
    PLACE_PATCH_2 = 2


# Then add the ActionSpace class
@dataclass
class ActionSpace:
    """Structured representation of the action space."""

    def __init__(self):
        """Initialize the action space with proper dimensions."""
        # Define the spaces for each action component
        self.patch_idx = gym.spaces.Discrete(4)  # -1 for pass, 0-2 for patches
        self.rotation = gym.spaces.Discrete(4)  # 0-3 for rotations
        self.row = gym.spaces.Discrete(9)  # 0-8 for board positions
        self.col = gym.spaces.Discrete(9)  # 0-8 for board positions

        # Combined space using MultiDiscrete
        self.space = gym.spaces.MultiDiscrete(
            [
                4,  # patch_idx (-1 to 2)
                4,  # rotation (0-3)
                9,  # row (0-8)
                9,  # col (0-8)
            ]
        )

    def sample(self):
        """Sample a random action from the action space."""
        return self.space.sample()

    def contains(self, x):
        """Check if an action is valid within this space."""
        return self.space.contains(x)


@dataclass
class ObservationSpace:
    """Structured representation of the observation space."""

    def __init__(self, board_size: int = 9, max_patches: int = 3):
        self.board_size = board_size
        self.max_patches = max_patches
        self.space = gym.spaces.Dict(
            {
                "player1_board": gym.spaces.Box(
                    low=0, high=1, shape=(board_size, board_size), dtype=np.int8
                ),
                "player2_board": gym.spaces.Box(
                    low=0, high=1, shape=(board_size, board_size), dtype=np.int8
                ),
                "player1_buttons": gym.spaces.Box(
                    low=0,
                    high=100,  # Reasonable max value instead of inf
                    shape=(1,),
                    dtype=np.float32,
                ),
                "player2_buttons": gym.spaces.Box(
                    low=0, high=100, shape=(1,), dtype=np.float32
                ),
                "player1_position": gym.spaces.Box(
                    low=0, high=53, shape=(1,), dtype=np.float32
                ),
                "player2_position": gym.spaces.Box(
                    low=0, high=53, shape=(1,), dtype=np.float32
                ),
                "current_player": gym.spaces.Discrete(2),
                "patch_shapes": gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(max_patches, 5, 5),  # Max patch size is 5x5
                    dtype=np.int8,
                ),
                "patch_buttons": gym.spaces.Box(
                    low=0,
                    high=20,  # Reasonable max value
                    shape=(max_patches,),
                    dtype=np.float32,
                ),
                "patch_times": gym.spaces.Box(
                    low=0,
                    high=10,  # Reasonable max value
                    shape=(max_patches,),
                    dtype=np.float32,
                ),
                "patch_incomes": gym.spaces.Box(
                    low=0,
                    high=5,  # Reasonable max value
                    shape=(max_patches,),
                    dtype=np.float32,
                ),
                "patch_rotations": gym.spaces.Box(
                    low=0, high=3, shape=(max_patches,), dtype=np.int8
                ),
                "patch_masks": gym.spaces.Box(
                    low=0, high=1, shape=(max_patches,), dtype=np.int8
                ),
            }
        )


class PatchworkGymEnv(gym.Env):
    """OpenAI Gym environment for the Patchwork game."""

    metadata = {"render_modes": ["console", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None):
        """Initialize the environment."""
        super().__init__()

        # Initialize game state and spaces
        self.state = PatchworkState()
        self.action_space = ActionSpace()
        self.observation_space = ObservationSpace()
        self.render_mode = render_mode

        # Initialize render-specific attributes
        if render_mode == "rgb_array":
            self._init_renderer()

    def _get_obs(self) -> Dict:
        """Convert current state to observation dictionary."""
        try:
            # Initialize patch arrays with zeros
            max_patches = self.observation_space.max_patches
            patch_shapes = np.zeros((max_patches, 5, 5), dtype=np.int8)
            patch_buttons = np.zeros(max_patches, dtype=np.float32)
            patch_times = np.zeros(max_patches, dtype=np.float32)
            patch_incomes = np.zeros(max_patches, dtype=np.float32)
            patch_rotations = np.zeros(max_patches, dtype=np.int8)
            patch_masks = np.zeros(max_patches, dtype=np.int8)

            # Fill in available patch data
            for i, patch in enumerate(self.state.available_patches[:max_patches]):
                # Pad patch shape to 5x5
                shape = patch.get_rotated_shape()
                h, w = shape.shape
                patch_shapes[i, :h, :w] = shape
                patch_buttons[i] = float(patch.buttons)
                patch_times[i] = float(patch.time)
                patch_incomes[i] = float(patch.button_income)
                patch_rotations[i] = patch.rotation
                patch_masks[i] = 1  # Mark this patch as valid

            return {
                "player1_board": np.asarray(self.state.player1_board, dtype=np.int8),
                "player2_board": np.asarray(self.state.player2_board, dtype=np.int8),
                "player1_buttons": np.array(
                    [float(self.state.player1_buttons)], dtype=np.float32
                ),
                "player2_buttons": np.array(
                    [float(self.state.player2_buttons)], dtype=np.float32
                ),
                "player1_position": np.array(
                    [float(self.state.player1_position)], dtype=np.float32
                ),
                "player2_position": np.array(
                    [float(self.state.player2_position)], dtype=np.float32
                ),
                "current_player": self.state.current_player - 1,  # Convert to 0/1
                "patch_shapes": patch_shapes,
                "patch_buttons": patch_buttons,
                "patch_times": patch_times,
                "patch_incomes": patch_incomes,
                "patch_rotations": patch_rotations,
                "patch_masks": patch_masks,
            }
        except Exception as e:
            logger.error(f"Error generating observation: {str(e)}")
            raise RuntimeError(f"Failed to generate observation: {str(e)}") from e

    def _get_info(self) -> Dict:
        """Return additional information about the current state."""
        return {
            "player1_score": self.state.get_score(1),
            "player2_score": self.state.get_score(2),
            "legal_moves": self.state.get_legal_moves(),
        }

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Reset game state
        self.state.reset()

        # Apply any custom reset options
        if options:
            self._apply_reset_options(options)

        try:
            # Get observation and info
            observation = self._get_obs()
            info = self._get_info()

            # Render if needed
            if self.render_mode == "console":
                self._render_console()

            return observation, info
        except Exception as e:
            logger.error(f"Error in reset: {str(e)}")
            raise RuntimeError(f"Environment reset failed: {str(e)}") from e

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        try:
            # Convert action to move
            move = self._action_to_move(action)

            # Execute move
            if self._is_valid_move(move):
                self.state.make_move(move)
            else:
                logger.warning(f"Invalid move attempted: {move}")
                # Execute pass move instead
                self.state.make_move(Move(None, None))

            # Get step results
            observation = self._get_obs()
            reward = self._calculate_reward()
            terminated = self.state.is_terminal()
            truncated = False
            info = self._get_info()

            # Update rendering if needed
            if self.render_mode == "console":
                self._render_console()

            return observation, reward, terminated, truncated, info

        except Exception as e:
            logger.error(f"Error in step execution: {str(e)}")
            raise

    def _calculate_reward(self) -> float:
        """Calculate the reward for the current state."""
        current_player = self.state.current_player
        score1 = self.state.get_score(1)
        score2 = self.state.get_score(2)

        if current_player == 1:
            return score1 - score2
        else:
            return score2 - score1

    def _action_to_move(self, action: np.ndarray) -> Move:
        """Convert gym action to game move."""
        patch_idx, rotation, row, col = action

        # Handle pass move
        if patch_idx == PatchworkAction.PASS:
            return Move(None, None)

        # Handle patch placement
        position = (int(row), int(col))
        return Move(int(patch_idx), position, int(rotation))

    def _is_valid_move(self, move: Move) -> bool:
        """Check if a move is valid."""
        return move in self.state.get_legal_moves()

    def render(self) -> Optional[np.ndarray]:
        """Render the current state."""
        if self.render_mode == "console":
            self._render_console()
            return None
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
        return None

    def _render_console(self) -> None:
        """Render the current state to console."""
        print("\nPlayer 1's Board:")
        self._print_board(self.state.player1_board)
        print("\nPlayer 2's Board:")
        self._print_board(self.state.player2_board)
        print(
            f"\nPlayer 1: {self.state.player1_buttons} buttons, "
            f"Position {self.state.player1_position}"
        )
        print(
            f"Player 2: {self.state.player2_buttons} buttons, "
            f"Position {self.state.player2_position}"
        )
        print(f"Current Player: {self.state.current_player}")

    def _print_board(self, board: np.ndarray) -> None:
        """Print a single board."""
        for row in board:
            print(" ".join("█" if cell else "." for cell in row))

    def _render_rgb_array(self) -> np.ndarray:
        """Render the state as an RGB array."""
        raise NotImplementedError("RGB array rendering not implemented yet")

    def close(self) -> None:
        """Clean up resources."""
        pass
