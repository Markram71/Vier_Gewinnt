"""Tests for neural network."""

import numpy as np
import torch

from src.network import ConnectFourNet
from src.config import BOARD_ROWS, BOARD_COLS


def test_forward_shape():
    """Output shapes are correct."""
    net = ConnectFourNet()
    batch_size = 8
    x = torch.randn(batch_size, 2, BOARD_ROWS, BOARD_COLS)
    policy, value = net(x)
    assert policy.shape == (batch_size, BOARD_COLS)
    assert value.shape == (batch_size, 1)


def test_predict_shape():
    """Single prediction returns correct shapes."""
    net = ConnectFourNet()
    board = np.random.randn(2, BOARD_ROWS, BOARD_COLS).astype(np.float32)
    policy, value = net.predict(board)
    assert policy.shape == (BOARD_COLS,)
    assert np.allclose(policy.sum(), 1.0)
    assert -1 <= value <= 1


def test_policy_valid_moves():
    """Policy can be masked for valid moves."""
    net = ConnectFourNet()
    board = np.zeros((2, BOARD_ROWS, BOARD_COLS), dtype=np.float32)
    board[0, 5, 0] = 1  # One stone in col 0
    policy, _ = net.predict(board)
    assert policy.shape == (BOARD_COLS,)
    assert np.all(policy >= 0)
