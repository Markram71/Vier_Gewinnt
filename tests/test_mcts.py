"""Tests for MCTS."""

import numpy as np
import torch

from src.game import ConnectFour
from src.mcts import MCTS
from src.network import ConnectFourNet


def test_mcts_returns_valid_policy():
    """MCTS returns normalized visit counts for valid moves."""
    net = ConnectFourNet()
    mcts = MCTS(network=net, num_simulations=20)
    game = ConnectFour()
    visit_counts = mcts.run(game)
    assert visit_counts.shape == (7,)
    assert np.allclose(visit_counts.sum(), 1.0)
    assert np.all(visit_counts >= 0)


def test_mcts_get_action():
    """MCTS returns valid action."""
    net = ConnectFourNet()
    mcts = MCTS(network=net, num_simulations=20)
    game = ConnectFour()
    action = mcts.get_action(game, temperature=0)
    assert action in game.get_valid_moves()


def test_mcts_plays_full_game():
    """MCTS can play a full game without errors."""
    net = ConnectFourNet()
    mcts = MCTS(network=net, num_simulations=25)
    game = ConnectFour()
    moves = 0
    while not game.is_terminal() and moves < 50:
        action = mcts.get_action(game, temperature=0.2)
        assert action in game.get_valid_moves()
        game.make_move(action)
        moves += 1
    assert game.is_terminal() or moves == 50
