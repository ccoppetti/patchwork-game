from typing import List, Tuple, Optional, Dict, Set, NamedTuple
import numpy as np
from dataclasses import dataclass
from patch_dataclass import Patch
from patch_config import (
    get_all_patches,
    get_starting_patch_order,
    GameConfig,
    get_leather_patch_positions,
    save_game_state,
    load_game_state,
)
import logging
from copy import deepcopy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlayerState(NamedTuple):
    """Represents the state of a single player."""

    board: np.ndarray
    buttons: int
    position: int
    button_income_spaces: Set[int]  # Tracks which income spaces have been counted


class Move(NamedTuple):
    """Represents a single move in the game."""

    patch_idx: Optional[int]
    position: Optional[Tuple[int, int]]
    rotation: int = 0

    @property
    def is_pass(self) -> bool:
        """Returns True if this is a passing move."""
        return self.patch_idx is None


class InvalidMoveError(Exception):
    """Raised when an invalid move is attempted."""

    pass


class GameState(NamedTuple):
    """Represents the complete state of a game for serialization."""

    player1: PlayerState
    player2: PlayerState
    current_player: int
    available_patches: List[Patch]
    patches_taken: Set[int]


class PatchworkState:
    """Represents the complete state of a Patchwork game."""

    def __init__(self):
        """Initialize a new game state."""
        self.config = GameConfig()
        self._initialize_game_state()
        logger.info("New game initialized")

    @property
    def player1_board(self):
        """Get player 1's board."""
        return self._player1.board

    @property
    def player2_board(self):
        """Get player 2's board."""
        return self._player2.board

    @property
    def player1_buttons(self):
        """Get player 1's button count."""
        return self._player1.buttons

    @property
    def player2_buttons(self):
        """Get player 2's button count."""
        return self._player2.buttons

    @property
    def player1_position(self):
        """Get player 1's position."""
        return self._player1.position

    @property
    def player2_position(self):
        """Get player 2's position."""
        return self._player2.position

    def _initialize_game_state(self) -> None:
        """Initialize or reset all game state variables."""
        board_shape = (self.config.BOARD_SIZE, self.config.BOARD_SIZE)
        self._player1 = PlayerState(
            board=np.zeros(board_shape, dtype=np.int8),
            buttons=self.config.INITIAL_BUTTONS,
            position=0,
            button_income_spaces=set(),
        )
        self._player2 = PlayerState(
            board=np.zeros(board_shape, dtype=np.int8),
            buttons=self.config.INITIAL_BUTTONS,
            position=0,
            button_income_spaces=set(),
        )
        self.current_player = 1
        self.available_patches = self._initialize_patches()
        self.patches_taken: Set[int] = set()
        self._leather_patches = get_leather_patch_positions()

    def _initialize_game_state(self) -> None:
        """Initialize or reset all game state variables."""
        board_shape = (self.config.BOARD_SIZE, self.config.BOARD_SIZE)
        self._player1 = PlayerState(
            board=np.zeros(board_shape, dtype=np.int8),
            buttons=self.config.INITIAL_BUTTONS,
            position=0,
            button_income_spaces=set(),
        )
        self._player2 = PlayerState(
            board=np.zeros(board_shape, dtype=np.int8),
            buttons=self.config.INITIAL_BUTTONS,
            position=0,
            button_income_spaces=set(),
        )
        self.current_player = 1
        self.available_patches = self._initialize_patches()
        self.patches_taken: Set[int] = set()
        self._leather_patches = get_leather_patch_positions()

    def _initialize_patches(self) -> List[Patch]:
        """Initialize all patches in their starting order."""
        try:
            all_patches = get_all_patches()
            logger.info(f"Successfully loaded {len(all_patches)} patches")

            patch_order = get_starting_patch_order()
            logger.info(
                f"Successfully loaded patch order with {len(patch_order)} positions"
            )

            if not all_patches:
                raise ValueError("No patches were loaded")

            if not patch_order:
                raise ValueError("Patch order is empty")

            # Validate patch IDs
            max_patch_id = len(all_patches)
            invalid_ids = [id for id in patch_order if id < 1 or id > max_patch_id]
            if invalid_ids:
                raise ValueError(f"Invalid patch IDs in order: {invalid_ids}")

            # Create ordered patch list
            ordered_patches = []
            for id in patch_order:
                try:
                    patch = all_patches[id - 1]
                    ordered_patches.append(patch)
                except IndexError:
                    raise ValueError(f"Invalid patch ID: {id}")

            logger.info(
                f"Successfully initialized {len(ordered_patches)} patches in game order"
            )
            return ordered_patches

        except Exception as e:
            logger.error(f"Failed to initialize patches: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            logger.error(f"Error details: {e.__dict__}")
            raise RuntimeError(f"Game initialization failed: {str(e)}") from e

    def get_legal_moves(self) -> List[Move]:
        """Returns list of all legal moves in current state."""
        legal_moves = []
        player = self._get_current_player()

        # Always can move forward (passing)
        legal_moves.append(Move(None, None))

        # Check next 3 patches for purchase possibilities
        for i in range(
            min(self.config.MAX_PATCHES_VISIBLE, len(self.available_patches))
        ):
            patch = self.available_patches[i]
            if patch.buttons <= player.buttons:
                # Try each rotation
                for rotation in range(4):
                    rotated_patch = patch.rotate(rotation)
                    valid_positions = self._find_valid_positions(rotated_patch)
                    for pos in valid_positions:
                        legal_moves.append(Move(i, pos, rotation))

        return legal_moves

    def _find_valid_positions(self, patch: Patch) -> List[Tuple[int, int]]:
        """Find all valid positions for a patch with its current rotation."""
        valid_positions = []
        shape = patch.get_rotated_shape()
        height, width = shape.shape
        board = self._get_current_player().board

        for i in range(self.config.BOARD_SIZE - height + 1):
            for j in range(self.config.BOARD_SIZE - width + 1):
                if self._is_valid_placement(board, shape, (i, j)):
                    valid_positions.append((i, j))

        return valid_positions

    def _is_valid_placement(
        self, board: np.ndarray, shape: np.ndarray, position: Tuple[int, int]
    ) -> bool:
        """Check if patch can be placed at given position."""
        i, j = position
        height, width = shape.shape

        # Check bounds
        if i + height > self.config.BOARD_SIZE or j + width > self.config.BOARD_SIZE:
            return False

        # Check overlap
        board_section = board[i : i + height, j : j + width]
        return not np.any(np.logical_and(board_section, shape))

    def make_move(self, move: Move) -> None:
        """Execute a move in the game.

        Args:
            move: Move to execute

        Raises:
            InvalidMoveError: If move is not legal
        """
        if move not in self.get_legal_moves():
            raise InvalidMoveError("Illegal move attempted")

        if move.is_pass:
            self._advance_time(1)
            return

        # Place patch
        patch = self.available_patches[move.patch_idx].rotate(move.rotation)
        player = self._get_current_player()

        try:
            self._place_patch(patch, move.position)
            self._update_player_state(patch)
            self.patches_taken.add(patch.id)
            self.available_patches.pop(move.patch_idx)
        except Exception as e:
            logger.error(f"Failed to execute move: {str(e)}")
            raise

    def _place_patch(self, patch: Patch, position: Tuple[int, int]) -> None:
        """Place a patch on the current player's board."""
        i, j = position
        shape = patch.get_rotated_shape()
        player = self._get_current_player()
        board_section = player.board[i : i + shape.shape[0], j : j + shape.shape[1]]
        player.board[i : i + shape.shape[0], j : j + shape.shape[1]] = np.logical_or(
            board_section, shape
        )

    def _update_player_state(self, patch: Patch) -> None:
        """Update player state after placing a patch."""
        player = self._get_current_player()
        self._advance_time(patch.time)
        self._update_buttons(patch)

    def _update_buttons(self, patch: Patch) -> None:
        """Update player's button count after placing a patch."""
        player = self._get_current_player()
        new_buttons = player.buttons - patch.buttons + patch.button_income
        if new_buttons < 0:
            raise InvalidMoveError("Player cannot afford this patch")
        self._set_player_buttons(new_buttons)

    def _advance_time(self, time_cost: int) -> None:
        """Advance the current player's position and handle income."""
        old_pos = self._get_current_player().position
        new_pos = min(old_pos + time_cost, self.config.TIME_TRACK_LENGTH)

        # Check each space crossed for income
        for pos in range(old_pos + 1, new_pos + 1):
            if pos % self.config.BUTTON_INCOME_INTERVAL == 0:
                self._grant_button_income(pos)
            if pos in self._leather_patches:
                self._grant_leather_patch(pos)

        # Update position and determine next player
        self._set_player_position(new_pos)
        self._update_current_player()

    def _grant_button_income(self, position: int) -> None:
        """Grant button income if space hasn't been counted yet."""
        player = self._get_current_player()
        if position not in player.button_income_spaces:
            income = self._calculate_button_income()
            self._set_player_buttons(player.buttons + income)
            player.button_income_spaces.add(position)

    def _calculate_button_income(self) -> int:
        """Calculate button income from placed patches."""
        player = self._get_current_player()
        return sum(
            patch.button_income
            for patch in self.available_patches
            if patch.id in self.patches_taken
        )

    def _grant_leather_patch(self, position: int) -> None:
        """Grant leather patch if available."""
        if position in self._leather_patches:
            shape = self._leather_patches[position]
            # TODO: Implement leather patch placement logic
            pass

    def _update_current_player(self) -> None:
        """Determine which player should move next."""
        p1_pos = self._player1.position
        p2_pos = self._player2.position
        self.current_player = 1 if p1_pos <= p2_pos else 2

    def _get_current_player(self) -> PlayerState:
        """Get the current player's state."""
        return self._player1 if self.current_player == 1 else self._player2

    def _set_player_buttons(self, buttons: int) -> None:
        """Update the current player's button count."""
        player = self._get_current_player()
        if self.current_player == 1:
            self._player1 = player._replace(buttons=buttons)
        else:
            self._player2 = player._replace(buttons=buttons)

    def _set_player_position(self, position: int) -> None:
        """Update the current player's position."""
        player = self._get_current_player()
        if self.current_player == 1:
            self._player1 = player._replace(position=position)
        else:
            self._player2 = player._replace(position=position)

    def get_score(self, player: int) -> int:
        """Calculate score for given player."""
        player_state = self._player1 if player == 1 else self._player2
        empty_squares = np.sum(player_state.board == 0) * -2
        return player_state.buttons + empty_squares

    def is_terminal(self) -> bool:
        """Check if game is over."""
        return (
            self._player1.position >= self.config.TIME_TRACK_LENGTH
            and self._player2.position >= self.config.TIME_TRACK_LENGTH
        )

    def get_observation(self) -> Dict:
        """Return current game state as observation."""
        return {
            "player1_board": self._player1.board.copy(),
            "player2_board": self._player2.board.copy(),
            "player1_buttons": self._player1.buttons,
            "player2_buttons": self._player2.buttons,
            "player1_position": self._player1.position,
            "player2_position": self._player2.position,
            "current_player": self.current_player,
            "available_patches": self.available_patches[
                : self.config.MAX_PATCHES_VISIBLE
            ],
        }

    def save_state(self, filename: str) -> None:
        """Save current game state to file."""
        state = GameState(
            player1=self._player1,
            player2=self._player2,
            current_player=self.current_player,
            available_patches=self.available_patches,
            patches_taken=self.patches_taken,
        )
        save_game_state(filename, state._asdict())

    def load_state(self, filename: str) -> None:
        """Load game state from file."""
        state = load_game_state(filename)
        game_state = GameState(**state)
        self._player1 = game_state.player1
        self._player2 = game_state.player2
        self.current_player = game_state.current_player
        self.available_patches = game_state.available_patches
        self.patches_taken = game_state.patches_taken

    def reset(self) -> None:
        """Reset the game to initial state."""
        self._initialize_game_state()
