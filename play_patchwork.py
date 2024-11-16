#!/usr/bin/env python3
import argparse
from pathlib import Path
from patchwork_cli import PatchworkCLI

def list_saves():
    """List all available save files."""
    saves = list(Path(".").glob("patchwork_save_*.json"))
    if not saves:
        print("No save files found.")
        return None
    
    print("\nAvailable save files:")
    for i, save in enumerate(saves):
        print(f"{i+1}. {save}")
    return saves

def load_game(save_file):
    """Load and start a game from a save file."""
    try:
        game = PatchworkCLI()
        game.game.load_state(save_file)
        print(f"\nLoaded save file: {save_file}")
        game.play_game()
    except Exception as e:
        print(f"Error loading save file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Patchwork Game Loader")
    parser.add_argument("--save", type=str, help="Save file to load")
    parser.add_argument("--new", action="store_true", help="Start new game")
    args = parser.parse_args()

    if args.new:
        game = PatchworkCLI()
        game.play_game()
        return

    if args.save:
        if Path(args.save).exists():
            load_game(args.save)
        else:
            print(f"Save file not found: {args.save}")
        return

    # If no arguments, show interactive menu
    saves = list_saves()
    if not saves:
        print("Starting new game...")
        game = PatchworkCLI()
        game.play_game()
        return

    while True:
        choice = input("\nEnter number to load game, 'n' for new game, or 'q' to quit: ")
        if choice.lower() == 'q':
            break
        elif choice.lower() == 'n':
            game = PatchworkCLI()
            game.play_game()
            break
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(saves):
                load_game(saves[idx])
                break
            else:
                print("Invalid selection.")
        except ValueError:
            print("Invalid input. Please enter a number, 'n', or 'q'.")

if __name__ == "__main__":
    main()
