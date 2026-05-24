"""Tests for self-play data generation."""

import numpy as np

from src.self_play import generate_self_play_data
from src.network import ConnectFourNet
from src.config import BOARD_ROWS, BOARD_COLS


def test_self_play_data_format():
    """Self-play produces correctly formatted training data."""
    net = ConnectFourNet()
    data = generate_self_play_data(net, num_games=2, num_simulations=10)
    assert len(data) > 0
    for state, policy, value in data:
        assert state.shape == (2, BOARD_ROWS, BOARD_COLS)
        assert policy.shape == (BOARD_COLS,)
        assert np.allclose(policy.sum(), 1.0)
        assert value in (-1.0, 0.0, 1.0)
