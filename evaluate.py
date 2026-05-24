#!/usr/bin/env python3
"""Evaluate two Connect Four checkpoints head-to-head.

Usage:
    python evaluate.py checkpoints/best_model.pt checkpoints/checkpoint_iter_480.pt
    python evaluate.py checkpoints/best_model.pt checkpoints/checkpoint_iter_480.pt --games 200
"""

import argparse
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from src.config import get_device
from src.game import ConnectFour
from src.mcts import MCTS
from src.network import ConnectFourNet


def load_checkpoint(path: Path, device: torch.device) -> ConnectFourNet:
    network = ConnectFourNet()
    state = torch.load(path, map_location=device, weights_only=False)
    network.load_state_dict(state.get("model", state))
    network.to(device)
    network.eval()
    return network


def play_one_game(mcts1: MCTS, mcts2: MCTS, mcts1_is_player1: bool) -> int:
    """
    Play one game between two agents.
    Returns +1 if mcts1 wins, -1 if mcts2 wins, 0 for draw.
    """
    mcts1_player = 1 if mcts1_is_player1 else -1
    game = ConnectFour()

    while not game.is_terminal():
        if game.current_player == mcts1_player:
            action = mcts1.get_action(game, temperature=0)
        else:
            action = mcts2.get_action(game, temperature=0)
        game.make_move(action)

    winner = game.check_win()
    if winner is None:
        return 0
    return 1 if winner == mcts1_player else -1


def evaluate(
    model1_path: Path,
    model2_path: Path,
    num_games: int = 100,
    num_simulations: int = 50,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Evaluate model1 against model2. Alternates who plays first for fairness.
    Returns win/draw/loss stats from model1's perspective.
    """
    if device is None:
        device = get_device()

    net1 = load_checkpoint(model1_path, device)
    net2 = load_checkpoint(model2_path, device)
    mcts1 = MCTS(net1, num_simulations=num_simulations, device=device)
    mcts2 = MCTS(net2, num_simulations=num_simulations, device=device)

    wins, losses, draws = 0, 0, 0

    for i in tqdm(range(num_games), desc="Evaluating"):
        # Alternate who plays as player 1 so neither side has a systematic advantage
        result = play_one_game(mcts1, mcts2, mcts1_is_player1=(i % 2 == 0))
        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            draws += 1

    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / num_games,
        "num_games": num_games,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate two Connect Four checkpoints head-to-head")
    parser.add_argument("model1", type=Path, help="Challenger model")
    parser.add_argument("model2", type=Path, help="Baseline model")
    parser.add_argument("--games", type=int, default=100, help="Number of games (default: 100)")
    parser.add_argument("--simulations", type=int, default=50, help="MCTS simulations per move (default: 50)")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu/cuda/mps)")
    args = parser.parse_args()

    device = get_device(args.device)
    print(f"Device:  {device}")
    print(f"Model 1: {args.model1}")
    print(f"Model 2: {args.model2}")
    print(f"Games:   {args.games}\n")

    stats = evaluate(args.model1, args.model2, args.games, args.simulations, device)

    print(f"\nResults (Model 1 vs Model 2):")
    print(f"  Wins:     {stats['wins']:4d} / {stats['num_games']}")
    print(f"  Draws:    {stats['draws']:4d} / {stats['num_games']}")
    print(f"  Losses:   {stats['losses']:4d} / {stats['num_games']}")
    print(f"  Win rate: {stats['win_rate']:.1%}")

    threshold = 0.55
    if stats["win_rate"] >= threshold:
        print(f"\n  Model 1 is stronger (≥{threshold:.0%} win rate).")
    else:
        print(f"\n  Model 2 holds its own (Model 1 win rate < {threshold:.0%}).")


if __name__ == "__main__":
    main()
