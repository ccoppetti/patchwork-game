import numpy as np
from typing import Dict, List, Tuple, NamedTuple
from dataclasses import dataclass
from patch_dataclass import Patch
import logging
import json
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GameConfig:
    """Game configuration parameters."""

    BOARD_SIZE: int = 9
    INITIAL_BUTTONS: int = 5
    TIME_TRACK_LENGTH: int = 53
    BUTTON_INCOME_INTERVAL: int = 6
    MAX_PATCHES_VISIBLE: int = 3


class PatchData(NamedTuple):
    """Data structure for patch definition."""

    shape: np.ndarray
    buttons: int
    time: int
    button_income: int
    color: Tuple[int, int, int]


# Define colors as RGB tuples with clear intention
PATCH_COLORS: Dict[str, Tuple[int, int, int]] = {
    "red": (220, 60, 50),  # Warm red
    "blue": (65, 105, 225),  # Royal blue
    "green": (34, 139, 34),  # Forest green
    "yellow": (255, 215, 0),  # Golden yellow
    "purple": (147, 112, 219),  # Medium purple
    "orange": (255, 140, 0),  # Dark orange
    "brown": (139, 69, 19),  # Saddle brown
    "pink": (255, 182, 193),  # Light pink
    "teal": (0, 128, 128),  # Teal
    "gray": (128, 128, 128),  # Medium gray
}


# Special tiles (leather patches) on the time track with their shapes
class LeatherPatch(NamedTuple):
    """Represents a leather patch on the time track."""

    position: int
    shape: Tuple[int, int]


LEATHER_PATCHES = [
    LeatherPatch(20, (1, 1)),  # Position 20: 1x1 leather patch
    LeatherPatch(26, (1, 1)),  # Position 26: 1x1 leather patch
    LeatherPatch(32, (1, 1)),  # Position 32: 1x1 leather patch
    LeatherPatch(44, (1, 1)),  # Position 44: 1x1 leather patch
    LeatherPatch(50, (1, 1)),  # Position 50: 1x1 leather patch
]


def load_patch_data() -> List[PatchData]:
    """Returns the standard patch configurations for the Patchwork game.

    Each patch is defined with its exact shape, button cost, time cost, and button income
    from the original board game.
    """
    try:
        patches = [
            # ID: 1 - Simple 2x2 square
            PatchData(
                shape=np.array([[1, 1], [1, 1]], dtype=np.int8),
                buttons=2,
                time=2,
                button_income=0,
                color=PATCH_COLORS["blue"],
            ),
            # ID: 2 - L shape
            PatchData(
                shape=np.array([[1, 0], [1, 0], [1, 1]], dtype=np.int8),
                buttons=3,
                time=2,
                button_income=0,
                color=PATCH_COLORS["green"],
            ),
            # ID: 3 - T shape
            PatchData(
                shape=np.array([[0, 1, 0], [1, 1, 1]], dtype=np.int8),
                buttons=2,
                time=2,
                button_income=0,
                color=PATCH_COLORS["red"],
            ),
            # ID: 4 - Long rectangle
            PatchData(
                shape=np.array([[1, 1, 1, 1]], dtype=np.int8),
                buttons=5,
                time=3,
                button_income=1,
                color=PATCH_COLORS["yellow"],
            ),
            # ID: 5 - Small L
            PatchData(
                shape=np.array([[1, 1], [0, 1]], dtype=np.int8),
                buttons=1,
                time=2,
                button_income=0,
                color=PATCH_COLORS["purple"],
            ),
            # ID: 6 - Corner piece
            PatchData(
                shape=np.array([[1, 1, 1], [1, 0, 0]], dtype=np.int8),
                buttons=3,
                time=2,
                button_income=0,
                color=PATCH_COLORS["orange"],
            ),
            # ID: 7 - Z shape
            PatchData(
                shape=np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8),
                buttons=4,
                time=2,
                button_income=1,
                color=PATCH_COLORS["teal"],
            ),
            # ID: 8 - Reverse L
            PatchData(
                shape=np.array([[1, 1], [1, 0], [1, 0]], dtype=np.int8),
                buttons=2,
                time=3,
                button_income=0,
                color=PATCH_COLORS["brown"],
            ),
            # ID: 9 - Wide U
            PatchData(
                shape=np.array([[1, 1, 1], [1, 0, 1]], dtype=np.int8),
                buttons=5,
                time=3,
                button_income=1,
                color=PATCH_COLORS["pink"],
            ),
            # ID: 10 - 3x2 rectangle
            PatchData(
                shape=np.array([[1, 1, 1], [1, 1, 1]], dtype=np.int8),
                buttons=6,
                time=5,
                button_income=2,
                color=PATCH_COLORS["gray"],
            ),
            # ID: 11 - Small T
            PatchData(
                shape=np.array([[1, 1, 1], [0, 1, 0]], dtype=np.int8),
                buttons=2,
                time=2,
                button_income=0,
                color=PATCH_COLORS["red"],
            ),
            # ID: 12 - Cross shape
            PatchData(
                shape=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int8),
                buttons=4,
                time=3,
                button_income=1,
                color=PATCH_COLORS["blue"],
            ),
            # ID: 13 - Large L
            PatchData(
                shape=np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=np.int8),
                buttons=4,
                time=4,
                button_income=1,
                color=PATCH_COLORS["green"],
            ),
            # ID: 14 - Small square with tail
            PatchData(
                shape=np.array([[1, 1, 1], [1, 1, 0]], dtype=np.int8),
                buttons=3,
                time=3,
                button_income=0,
                color=PATCH_COLORS["yellow"],
            ),
            # ID: 15 - Long L
            PatchData(
                shape=np.array([[1, 0, 0, 0], [1, 1, 1, 1]], dtype=np.int8),
                buttons=5,
                time=4,
                button_income=1,
                color=PATCH_COLORS["purple"],
            ),
            # For now, completing with placeholder 2x2 squares for testing
            # Replace these with actual patch configurations
            *[
                PatchData(
                    shape=np.array([[1, 1], [1, 1]], dtype=np.int8),
                    buttons=2,
                    time=2,
                    button_income=0,
                    color=PATCH_COLORS[
                        list(PATCH_COLORS.keys())[i % len(PATCH_COLORS)]
                    ],
                )
                for i in range(15, 33)
            ],
        ]

        logger.info(f"Successfully loaded {len(patches)} patch configurations")
        if len(patches) < 33:
            raise ValueError(
                f"Not enough patches defined. Expected 33, got {len(patches)}"
            )
        return patches

    except Exception as e:
        logger.error(f"Failed to load patch data: {str(e)}")
        raise RuntimeError(f"Failed to load patch data: {str(e)}") from e

    except Exception as e:
        logger.error(f"Failed to load patch data: {str(e)}")
        raise RuntimeError(f"Failed to load patch data: {str(e)}") from e


def create_patch(patch_data: PatchData, patch_id: int) -> Patch:
    """Create a Patch instance from patch data.

    Args:
        patch_data: PatchData containing shape and properties
        patch_id: Unique identifier for the patch

    Returns:
        New Patch instance

    Raises:
        ValueError: If patch data is invalid
    """
    try:
        return Patch.from_array(
            shape_array=patch_data.shape,
            buttons=patch_data.buttons,
            time=patch_data.time,
            button_income=patch_data.button_income,
            id=patch_id,
            color=patch_data.color,
        )
    except ValueError as e:
        logger.error(f"Failed to create patch {patch_id}: {str(e)}")
        raise


def get_all_patches() -> List[Patch]:
    """Returns all patches in the standard Patchwork game.

    Returns:
        List of Patch instances representing all game pieces

    Raises:
        RuntimeError: If patch creation fails
    """
    patches = []
    patch_data = load_patch_data()

    for idx, data in enumerate(patch_data, 1):
        try:
            patch = create_patch(data, idx)
            patches.append(patch)
        except ValueError as e:
            logger.error(f"Error creating patch {idx}: {str(e)}")
            raise RuntimeError(f"Failed to initialize game pieces: {str(e)}")

    return patches


def get_starting_patch_order() -> List[int]:
    """Returns the standard initial ordering of patches on the time board.

    These are patch IDs in the order they appear on the time track.

    Returns:
        List of patch IDs in their starting order
    """
    return [
        2,
        4,
        1,
        7,
        3,
        12,
        5,
        9,
        14,
        6,
        19,
        8,
        24,
        11,
        27,
        13,
        29,
        15,
        31,
        16,
        32,
        17,
        33,
        18,
        30,
        20,
        28,
        21,
        26,
        22,
        25,
        23,
        10,
    ]


def get_leather_patch_positions() -> Dict[int, Tuple[int, int]]:
    """Returns a mapping of time track positions to leather patch shapes.

    Returns:
        Dictionary mapping position to shape tuple
    """
    return {patch.position: patch.shape for patch in LEATHER_PATCHES}


def save_game_state(filename: str, game_state: dict) -> None:
    """Save game state to a JSON file.

    Args:
        filename: Path to save file
        game_state: Dictionary containing game state

    Raises:
        IOError: If save fails
    """
    try:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy arrays to lists for JSON serialization
        serializable_state = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in game_state.items()
        }

        with open(path, "w") as f:
            json.dump(serializable_state, f)
    except Exception as e:
        logger.error(f"Failed to save game state: {str(e)}")
        raise


def load_game_state(filename: str) -> dict:
    """Load game state from a JSON file.

    Args:
        filename: Path to save file

    Returns:
        Dictionary containing game state

    Raises:
        IOError: If load fails
        ValueError: If file contains invalid data
    """
    try:
        with open(filename, "r") as f:
            state = json.load(f)

        # Convert lists back to numpy arrays where needed
        for key in ["player1_board", "player2_board"]:
            if key in state:
                state[key] = np.array(state[key], dtype=np.int8)

        return state
    except Exception as e:
        logger.error(f"Failed to load game state: {str(e)}")
        raise
