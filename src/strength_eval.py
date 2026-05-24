"""Lightweight strength evaluation against simple baseline agents."""

import random
from typing import Optional

import numpy as np
import torch

from .game import ConnectFour
from .mcts import MCTS
from .network import ConnectFourNet


class RandomAgent:
    """Plays a uniformly random valid move."""

    def get_action(self, game: ConnectFour, temperature: float = 0.0) -> int:
        return random.choice(game.get_valid_moves())


class HeuristicAgent:
    """Wins immediately if possible, blocks opponent wins, otherwise random.

    Matches the forced-move logic used inside MCTS, making it a meaningful
    baseline: a well-trained model should beat it consistently.
    """

    def get_action(self, game: ConnectFour, temperature: float = 0.0) -> int:
        win = game.get_winning_move()
        if win is not None:
            return win
        block = game.get_blocking_move()
        if block is not None:
            return block
        return random.choice(game.get_valid_moves())


def _play_one_game(mcts: MCTS, opponent, mcts_is_player1: bool) -> int:
    """
    Play a single game between an MCTS agent and any agent with get_action().
    Returns +1 if MCTS wins, -1 if opponent wins, 0 for draw.
    """
    mcts_player = 1 if mcts_is_player1 else -1
    game = ConnectFour()
    while not game.is_terminal():
        if game.current_player == mcts_player:
            action = mcts.get_action(game, temperature=0)
        else:
            action = opponent.get_action(game)
        game.make_move(action)
    winner = game.check_win()
    if winner is None:
        return 0
    return 1 if winner == mcts_player else -1


def eval_strength(
    network: ConnectFourNet,
    n_games: int = 20,
    num_simulations: int = 20,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Evaluate network win rate against RandomAgent and HeuristicAgent.

    Uses a low simulation count for speed during training.
    Alternates who plays first so neither side has a systematic advantage.

    Returns:
        dict with keys: vs_random, vs_heuristic (both as float win rates 0–1)
    """
    if device is None:
        device = next(network.parameters()).device

    mcts = MCTS(network, num_simulations=num_simulations, device=device)
    random_agent = RandomAgent()
    heuristic_agent = HeuristicAgent()

    results = {}
    for label, opponent in (("vs_random", random_agent), ("vs_heuristic", heuristic_agent)):
        wins = 0
        for i in range(n_games):
            outcome = _play_one_game(mcts, opponent, mcts_is_player1=(i % 2 == 0))
            if outcome == 1:
                wins += 1
        results[label] = wins / n_games

    return results
