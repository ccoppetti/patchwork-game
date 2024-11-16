from typing import List, Optional, Tuple, Dict
import numpy as np
from patch_dataclass import Patch
from dataclasses import dataclass
import os
import platform
from enum import Enum

class Color:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"
    GRAY = "\033[90m"

    @staticmethod
    def rgb_to_ansi(r: int, g: int, b: int) -> str:
        """Convert RGB color to closest ANSI color code."""
        colors = {
            Color.RED: (255, 0, 0),
            Color.GREEN: (0, 255, 0),
            Color.BLUE: (0, 0, 255),
            Color.YELLOW: (255, 255, 0),
            Color.MAGENTA: (255, 0, 255),
            Color.CYAN: (0, 255, 255),
            Color.WHITE: (255, 255, 255),
            Color.GRAY: (128, 128, 128)
        }
        
        def color_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
            return sum((x - y) ** 2 for x, y in zip(c1, c2))
        
        closest = min(colors.items(), key=lambda x: color_distance((r, g, b), x[1]))
        return closest[0]

class BoardSymbols(Enum):
    """Unicode symbols for board display."""
    EMPTY = "·"
    FILLED = "█"
    PREVIEW = "▒"
    BORDER_H = "─"
    BORDER_V = "│"
    CORNER_TL = "┌"
    CORNER_TR = "┐"
    CORNER_BL = "└"
    CORNER_BR = "┘"

@dataclass
class DisplayConfig:
    """Configuration for display formatting."""
    board_size: int
    cell_width: int = 2
    show_colors: bool = True

    def __post_init__(self):
        self.show_colors = self._check_color_support()

    @staticmethod
    def _check_color_support() -> bool:
        """Check if terminal supports color output."""
        return (platform.system() != "Windows" or
                os.environ.get("ANSICON") is not None or
                os.environ.get("WT_SESSION") is not None)

class PatchworkDisplay:
    """Handles all display-related functionality for the Patchwork game."""

    def __init__(self, board_size: int):
        self.config = DisplayConfig(board_size)

    def clear_screen(self) -> None:
        """Clear the terminal screen in a cross-platform way."""
        os.system('cls' if platform.system() == 'Windows' else 'clear')

    def display_board(self, board: np.ndarray, player: int, title: Optional[str] = None) -> None:
        """Display a player's board with nice formatting.
        
        Args:
            board: The game board to display
            player: Player number (1 or 2)
            title: Optional title to display above board
        """
        # Display title
        if title:
            print(f"\n{Color.BOLD}{title}{Color.RESET}")
        else:
            print(f"\n{Color.BOLD}Player {player}'s Board:{Color.RESET}")

        # Display column numbers
        print("   ", end="")
        for i in range(self.config.board_size):
            print(f"{i:2}", end="")
        print()

        # Display top border
        print(f"  {BoardSymbols.CORNER_TL.value}", end="")
        print(BoardSymbols.BORDER_H.value * (self.config.board_size * 2), end="")
        print(BoardSymbols.CORNER_TR.value)

        # Display board rows
        for i in range(self.config.board_size):
            print(f"{i:2}{BoardSymbols.BORDER_V.value}", end="")
            for j in range(self.config.board_size):
                symbol = (BoardSymbols.FILLED.value if board[i, j]
                         else BoardSymbols.EMPTY.value)
                print(f"{symbol} ", end="")
            print(BoardSymbols.BORDER_V.value)

        # Display bottom border
        print(f"  {BoardSymbols.CORNER_BL.value}", end="")
        print(BoardSymbols.BORDER_H.value * (self.config.board_size * 2), end="")
        print(BoardSymbols.CORNER_BR.value)

    def display_patch(self, patch: Patch, index: Optional[int] = None) -> None:
        """Display a single patch with its properties and current rotation."""
        if index is not None:
            print(f"\n{Color.BOLD}Patch {index}:{Color.RESET}")
        
        # Get color for patch display
        color = Color.rgb_to_ansi(*patch.color) if self.config.show_colors else ""
        
        # Display shape
        print("\nShape (current rotation):")
        rotated_shape = patch.get_rotated_shape()
        for row in rotated_shape:
            print("  ", end="")
            for cell in row:
                symbol = BoardSymbols.FILLED.value if cell else BoardSymbols.EMPTY.value
                print(f"{color}{symbol}{Color.RESET} ", end="")
            print()

        # Display properties
        print(f"Cost: {patch.buttons} buttons")
        print(f"Time: {patch.time} spaces")
        print(f"Income: {patch.button_income} buttons")
        print(f"Current rotation: {patch.rotation * 90}°")

    def display_patch_preview(
        self,
        current_board: np.ndarray,
        patch: Patch,
        position: Tuple[int, int],
        valid: bool
    ) -> None:
        """Display a preview of where the patch would be placed."""
        print(f"\n{Color.BOLD}Placement Preview:{Color.RESET}")
        
        i, j = position
        rotated_shape = patch.get_rotated_shape()
        height, width = rotated_shape.shape
        
        # Create preview board
        preview_board = current_board.copy()
        if valid and i + height <= self.config.board_size and j + width <= self.config.board_size:
            preview_board[i:i + height, j:j + width] |= rotated_shape
            
            # Display the preview
            self.display_board_with_preview(
                current_board,
                preview_board,
                Color.rgb_to_ansi(*patch.color) if self.config.show_colors else ""
            )
        else:
            print(f"{Color.RED}Invalid placement - out of bounds or overlapping!{Color.RESET}")

    def display_board_with_preview(
        self,
        current_board: np.ndarray,
        preview_board: np.ndarray,
        color: str
    ) -> None:
        """Display board with preview overlay."""
        # Column numbers
        print("   ", end="")
        for i in range(self.config.board_size):
            print(f"{i:2}", end="")
        print()

        # Top border
        print(f"  {BoardSymbols.CORNER_TL.value}", end="")
        print(BoardSymbols.BORDER_H.value * (self.config.board_size * 2), end="")
        print(BoardSymbols.CORNER_TR.value)

        # Board content
        for i in range(self.config.board_size):
            print(f"{i:2}{BoardSymbols.BORDER_V.value}", end="")
            for j in range(self.config.board_size):
                if preview_board[i, j] == 1:
                    if current_board[i, j] == 1:
                        print(f"{BoardSymbols.FILLED.value} ", end="")
                    else:
                        print(f"{color}{BoardSymbols.PREVIEW.value}{Color.RESET} ", end="")
                else:
                    print(f"{BoardSymbols.EMPTY.value} ", end="")
            print(BoardSymbols.BORDER_V.value)

        # Bottom border
        print(f"  {BoardSymbols.CORNER_BL.value}", end="")
        print(BoardSymbols.BORDER_H.value * (self.config.board_size * 2), end="")
        print(BoardSymbols.CORNER_BR.value)

    def display_game_stats(
        self,
        player1_buttons: int,
        player2_buttons: int,
        player1_position: int,
        player2_position: int,
        current_player: int
    ) -> None:
        """Display current game statistics."""
        print(f"\n{Color.BOLD}Game Status:{Color.RESET}")
        p1_color = Color.CYAN if current_player == 1 else Color.RESET
        p2_color = Color.CYAN if current_player == 2 else Color.RESET
        
        print(f"{p1_color}Player 1: {player1_buttons} buttons, "
              f"Position {player1_position}{Color.RESET}")
        print(f"{p2_color}Player 2: {player2_buttons} buttons, "
              f"Position {player2_position}{Color.RESET}")
        print(f"\nCurrent Player: {current_player}")

    def display_scores(self, player1_score: int, player2_score: int) -> None:
        """Display final game scores."""
        print(f"\n{Color.BOLD}Game Over!{Color.RESET}")
        print(f"Player 1 score: {player1_score}")
        print(f"Player 2 score: {player2_score}")
        
        if player1_score > player2_score:
            print(f"{Color.GREEN}Player 1 wins!{Color.RESET}")
        elif player2_score > player1_score:
            print(f"{Color.GREEN}Player 2 wins!{Color.RESET}")
        else:
            print(f"{Color.YELLOW}It's a tie!{Color.RESET}")

    def display_welcome(self) -> None:
        """Display welcome message and game instructions."""
        print(f"\n{Color.BOLD}Welcome to Patchwork!{Color.RESET}")
        print("Each player starts with 5 buttons and an empty 9x9 board.")
        print("Your goal is to create the most valuable quilt by buying and")
        print("placing patches while managing your buttons and time.")
        print("\nPress Enter to begin...")
        input()
