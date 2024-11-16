from typing import Optional, Tuple
import numpy as np
from patchwork_env import PatchworkState, Move, InvalidMoveError
from patch_display import PatchworkDisplay, Color
import logging
from pathlib import Path
import json
import sys
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("patchwork.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class PatchworkCLI:
    """Command-line interface for the Patchwork game."""

    def __init__(self):
        """Initialize game state and display manager."""
        self.game = PatchworkState()
        self.display = PatchworkDisplay(self.game.config.BOARD_SIZE)
        self.autosave_enabled = True

    def get_player_move(self) -> Move:
        """Get and validate move input from the current player.

        Returns:
            Move: Valid move to be executed

        Raises:
            KeyboardInterrupt: If user requests to quit
        """
        while True:
            try:
                print(f"\n{Color.BOLD}Options:{Color.RESET}")
                print("1. Buy and place a patch")
                print("2. Pass (move forward)")
                print("3. Save game")
                print("4. Quit")

                choice = input(f"\n{Color.CYAN}Enter your choice (1-4): {Color.RESET}")

                if choice == "4":
                    raise KeyboardInterrupt
                elif choice == "3":
                    self._handle_save_game()
                    continue
                elif choice == "2":
                    return Move(None, None)
                elif choice == "1":
                    return self._handle_patch_placement()
                else:
                    print(f"{Color.RED}Invalid choice! Please enter 1-4.{Color.RESET}")

            except ValueError as e:
                print(f"{Color.RED}Invalid input! {str(e)}{Color.RESET}")
            except Exception as e:
                logger.error(f"Unexpected error in get_player_move: {str(e)}")
                print(f"{Color.RED}An error occurred. Please try again.{Color.RESET}")

    def _handle_patch_placement(self) -> Move:
        """Handle the patch selection and placement process.

        Returns:
            Move: Valid patch placement move
        """
        # Select patch
        patch_idx = self._get_patch_selection()
        original_patch = self.game.available_patches[patch_idx]

        # Handle rotation
        total_rotation = self._get_patch_rotation(original_patch)
        rotated_patch = original_patch.rotate(total_rotation)

        # Get placement position
        position = self._get_placement_position(rotated_patch)

        return Move(patch_idx, position, total_rotation)

    def _get_patch_selection(self) -> int:
        """Get valid patch selection from user.

        Returns:
            int: Index of selected patch
        """
        while True:
            try:
                idx = int(input(f"{Color.CYAN}Enter patch number (0-2): {Color.RESET}"))
                if 0 <= idx <= 2 and idx < len(self.game.available_patches):
                    return idx
                print(f"{Color.RED}Invalid patch number!{Color.RESET}")
            except ValueError:
                print(f"{Color.RED}Please enter a number!{Color.RESET}")

    def _get_patch_rotation(self, original_patch) -> int:
        """Get valid rotation selection from user.

        Args:
            original_patch: The patch to rotate

        Returns:
            int: Total number of 90-degree rotations to apply
        """
        total_rotation = 0
        current_patch = original_patch

        while True:
            self.display.display_patch(current_patch)
            print("\nRotation options:")
            print("1. Rotate clockwise 90°")
            print("2. Proceed with current rotation")

            choice = input(f"{Color.CYAN}Enter choice (1-2): {Color.RESET}")

            if choice == "1":
                total_rotation = (total_rotation + 1) % 4  # Keep rotations between 0-3
                current_patch = original_patch.rotate(total_rotation)
                continue
            elif choice == "2":
                return total_rotation
            print(f"{Color.RED}Invalid choice!{Color.RESET}")

    def _get_placement_position(self, patch) -> Tuple[int, int]:
        """Get valid placement position from user.

        Args:
            patch: The rotated patch to place

        Returns:
            Tuple[int, int]: (row, column) position for placement
        """
        while True:
            try:
                row = int(input(f"{Color.CYAN}Enter row: {Color.RESET}"))
                col = int(input(f"{Color.CYAN}Enter column: {Color.RESET}"))
                position = (row, col)

                # Show preview and get confirmation
                current_board = (
                    self.game.player1_board
                    if self.game.current_player == 1
                    else self.game.player2_board
                )

                valid = self.game._is_valid_placement(
                    current_board, patch.get_rotated_shape(), position
                )

                self.display.display_patch_preview(
                    current_board, patch, position, valid
                )

                if not valid:
                    continue

                if (
                    input(
                        f"\n{Color.CYAN}Confirm placement? (y/n): {Color.RESET}"
                    ).lower()
                    == "y"
                ):
                    return position

            except ValueError:
                print(f"{Color.RED}Please enter valid numbers!{Color.RESET}")

    def _handle_save_game(self) -> None:
        """Handle game save functionality."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"patchwork_save_{timestamp}.json"
            self.game.save_state(filename)
            print(f"{Color.GREEN}Game saved to {filename}{Color.RESET}")
        except Exception as e:
            logger.error(f"Failed to save game: {str(e)}")
            print(f"{Color.RED}Failed to save game: {str(e)}{Color.RESET}")

    def _autosave(self) -> None:
        """Automatically save game state."""
        if self.autosave_enabled:
            try:
                self.game.save_state("patchwork_autosave.json")
            except Exception as e:
                logger.error(f"Autosave failed: {str(e)}")

    def play_game(self) -> None:
        """Main game loop."""
        try:
            self.display.display_welcome()

            while not self.game.is_terminal():
                self.display.clear_screen()
                self._display_game_state()

                try:
                    move = self.get_player_move()
                    self.game.make_move(move)
                    self._autosave()
                except InvalidMoveError as e:
                    print(f"{Color.RED}Invalid move: {str(e)}{Color.RESET}")
                    input("Press Enter to continue...")
                except KeyboardInterrupt:
                    if (
                        input(
                            f"{Color.YELLOW}Save game before quitting? (y/n): {Color.RESET}"
                        ).lower()
                        == "y"
                    ):
                        self._handle_save_game()
                    print("\nThanks for playing!")
                    return

            # Game over - display final state
            self.display.clear_screen()
            self._display_game_state()
            self._display_final_scores()

        except Exception as e:
            logger.error(f"Unexpected error in play_game: {str(e)}")
            print(f"{Color.RED}An error occurred: {str(e)}{Color.RESET}")
            print("The game state has been autosaved.")

    def _display_game_state(self) -> None:
        """Display the complete current game state."""
        # Display boards
        self.display.display_board(self.game.player1_board, 1)
        self.display.display_board(self.game.player2_board, 2)

        # Display game stats
        self.display.display_game_stats(
            self.game.player1_buttons,
            self.game.player2_buttons,
            self.game.player1_position,
            self.game.player2_position,
            self.game.current_player,
        )

        # Display available patches
        print(f"\n{Color.BOLD}Available Patches:{Color.RESET}")
        for i, patch in enumerate(self.game.available_patches[:3]):
            self.display.display_patch(patch, i)

    def _display_final_scores(self) -> None:
        """Display final game scores and winner."""
        score1 = self.game.get_score(1)
        score2 = self.game.get_score(2)
        self.display.display_scores(score1, score2)


def main():
    """Entry point for the game."""
    try:
        game = PatchworkCLI()
        game.play_game()
    except KeyboardInterrupt:
        print("\nGame terminated by user.")
    except Exception as e:
        logger.error(f"Critical error in main: {str(e)}")
        print(
            f"{Color.RED}A critical error occurred. Check patchwork.log for details.{Color.RESET}"
        )
    finally:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
