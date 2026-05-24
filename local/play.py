"""Local terminal game for playing Connect Four against the trained AI."""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BOARD_ROWS, BOARD_COLS, get_device
from src.game import ConnectFour
from src.mcts import MCTS
from src.network import ConnectFourNet

console = Console()


def load_model(checkpoint_path: Path, device: torch.device, num_simulations: int) -> MCTS:
    network = ConnectFourNet()
    if checkpoint_path.exists():
        try:
            state = torch.load(checkpoint_path, map_location=device, weights_only=False)
            network.load_state_dict(state.get("model", state))
            console.print(f"[green]Loaded model from {checkpoint_path}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to load model: {e}. Using untrained network.[/red]")
    else:
        console.print(f"[yellow]No checkpoint at {checkpoint_path}, using untrained network[/yellow]")
    network.to(device)
    network.eval()
    return MCTS(network=network, num_simulations=num_simulations, device=device)


def render_board(
    game: ConnectFour,
    human_player: int,
    winning_line: Optional[list] = None,
) -> Text:
    """Render the board: human pieces are red, AI pieces are yellow."""
    winning_set = set(map(tuple, winning_line)) if winning_line else set()
    text = Text()
    for row in range(BOARD_ROWS):
        for col in range(BOARD_COLS):
            val = game.board[row, col]
            pos = (row, col)
            if val == human_player:
                cell = Text("● ", style="bold red")
            elif val == -human_player:
                cell = Text("● ", style="bold yellow")
            else:
                cell = Text("· ", style="dim gray")
            if pos in winning_set:
                cell = Text("● ", style="bold white on blue")
            text.append_text(cell)
        text.append("\n")
    text.append(" ")
    for c in range(BOARD_COLS):
        text.append(f"{c} ", style="dim")
    return text


def play_game(mcts: MCTS, human_first: bool) -> None:
    """Run the main game loop: human vs AI."""
    human_player = 1 if human_first else -1
    ai_player = -human_player

    game = ConnectFour()

    who_str = "You go first (red ●)" if human_first else "AI goes first (red ●) — you are yellow ●"
    console.print(Panel(f"[bold]Connect Four[/bold] — {who_str}", expand=False))
    console.print("Enter column 0–6 to drop a piece. Type 'q' to quit.\n")

    while True:
        console.print(render_board(game, human_player))
        valid = game.get_valid_moves()

        if game.is_terminal():
            winner = game.check_win()
            line = game.get_winning_line()
            console.print(render_board(game, human_player, winning_line=line))
            if winner == human_player:
                console.print(Panel("[bold green]You win![/bold green]", expand=False))
            elif winner == ai_player:
                console.print(Panel("[bold red]AI wins![/bold red]", expand=False))
            else:
                console.print(Panel("[bold yellow]Draw![/bold yellow]", expand=False))
            break

        if game.current_player == human_player:
            console.print("[bold red]Your turn.[/bold red] Valid columns:", valid)
            try:
                inp = input("Column (0–6): ").strip().lower()
                if inp == "q":
                    console.print("Goodbye!")
                    return
                col = int(inp)
                if col not in valid:
                    console.print(f"[red]Invalid move. Choose from {valid}[/red]\n")
                    continue
            except ValueError:
                console.print("[red]Enter a number 0–6 or 'q' to quit[/red]\n")
                continue
            game.make_move(col)
        else:
            console.print("[yellow]AI thinking...[/yellow]")
            action = mcts.get_action(game, temperature=0)
            game.make_move(action)
            console.print(f"[yellow]AI plays column {action}[/yellow]\n")


def main():
    parser = argparse.ArgumentParser(description="Play Connect Four vs AI in the terminal")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "checkpoints" / "best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=50,
        help="MCTS simulations per move (default: 50)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (cpu/cuda/mps). Auto-detected if not set.",
    )
    parser.add_argument(
        "--first",
        choices=["human", "ai"],
        default="human",
        help="Who plays first (default: human)",
    )
    args = parser.parse_args()

    device = get_device(args.device)
    console.print(f"[dim]Device: {device}[/dim]")

    mcts = load_model(args.checkpoint, device, args.simulations)
    play_game(mcts, human_first=(args.first == "human"))


if __name__ == "__main__":
    main()
