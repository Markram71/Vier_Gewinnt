"""Unit tests for Connect Four game logic."""

import numpy as np
import pytest

from src.game import ConnectFour
from src.config import BOARD_ROWS, BOARD_COLS


def test_initial_state():
    """Empty board, player 1 to move."""
    game = ConnectFour()
    assert game.board.shape == (BOARD_ROWS, BOARD_COLS)
    assert np.all(game.board == 0)
    assert game.current_player == 1
    assert game.get_valid_moves() == list(range(BOARD_COLS))


def test_valid_moves():
    """Valid moves shrink as columns fill."""
    game = ConnectFour()
    game.make_move(3)
    game.make_move(3)
    game.make_move(3)
    game.make_move(3)
    game.make_move(3)
    game.make_move(3)
    moves = game.get_valid_moves()
    assert 3 not in moves
    assert len(moves) == BOARD_COLS - 1


def test_make_move():
    """Stones drop to bottom of column."""
    game = ConnectFour()
    assert game.make_move(0)
    assert game.board[BOARD_ROWS - 1, 0] == 1
    assert game.current_player == -1

    game.make_move(0)
    assert game.board[BOARD_ROWS - 2, 0] == -1
    assert game.current_player == 1


def test_invalid_move():
    """Invalid column returns False."""
    game = ConnectFour()
    assert not game.make_move(-1)
    assert not game.make_move(BOARD_COLS)
    # Fill column 0 completely, then it becomes invalid
    for _ in range(BOARD_ROWS):
        game.make_move(0)
        if not game.is_terminal():
            game.make_move(1)
    assert not game.make_move(0)


def test_horizontal_win():
    """Four in a row horizontally."""
    game = ConnectFour()
    for col in [0, 1, 2, 3]:
        game.make_move(col)
        game.make_move(6)
    assert game.check_win() == 1


def test_vertical_win():
    """Four in a column."""
    game = ConnectFour()
    for _ in range(4):
        game.make_move(0)
        game.make_move(1)
    assert game.check_win() == 1


def test_diagonal_win_down_right():
    """Diagonal down-right."""
    game = ConnectFour()
    moves = [(0, 0), (1, 1), (2, 2), (3, 3)]
    for i, (p1_col, p2_col) in enumerate(moves):
        game.make_move(p1_col)
        if not game.is_terminal():
            game.make_move(p2_col)
    assert game.check_win() == 1


def test_diagonal_win_down_left():
    """Diagonal down-left: (5,3), (4,4), (3,5), (2,6)."""
    game = ConnectFour()
    # Build diagonal: P1 needs (5,3), (4,4), (3,5), (2,6)
    # P2 blocks elsewhere (alternate cols 0,1,2 to avoid filling)
    p2_cols = [0, 1, 2] * 4
    game.make_move(3)
    for i in range(2):
        game.make_move(4)
        game.make_move(p2_cols[i])
    for i in range(2, 5):
        game.make_move(5)
        game.make_move(p2_cols[i])
    for i in range(5, 9):
        game.make_move(6)
        game.make_move(p2_cols[i])
    assert game.check_win() == 1


def test_draw():
    """is_draw returns False when moves exist or when there is a winner."""
    game = ConnectFour()
    assert not game.is_draw()
    game.make_move(0)
    assert not game.is_draw()
    # Win is not a draw
    game = ConnectFour()
    for col in [0, 1, 2, 3]:
        game.make_move(col)
        if not game.is_terminal():
            game.make_move(6)
    assert game.check_win() == 1
    assert not game.is_draw()


def test_clone():
    """Clone is independent copy."""
    game = ConnectFour()
    game.make_move(0)
    game.make_move(1)
    clone = game.clone()
    clone.make_move(2)
    assert game.board[BOARD_ROWS - 1, 2] == 0
    assert clone.board[BOARD_ROWS - 1, 2] == 1


def test_to_tensor_shape():
    """Tensor has correct shape."""
    game = ConnectFour()
    t = game.to_tensor()
    assert t.shape == (2, BOARD_ROWS, BOARD_COLS)


def test_to_tensor_canonical():
    """Canonical form swaps channels for player -1."""
    game = ConnectFour()
    game.make_move(0)
    game.current_player = 1
    t1 = game.to_tensor(canonical=True)
    game.current_player = -1
    t2 = game.to_tensor(canonical=True)
    assert np.allclose(t1[0], t2[1])
    assert np.allclose(t1[1], t2[0])


def test_get_winning_move():
    """Winning move returns column that wins immediately."""
    game = ConnectFour()
    # Build three in a row horizontally for P1: cols 0,1,2
    for col in [0, 1, 2]:
        game.make_move(col)
        game.make_move(6)  # P2 blocks elsewhere
    assert game.get_winning_move() == 3  # P1 wins with col 3

    game = ConnectFour()
    # P1 gets three in col 3, P2 plays elsewhere
    game.make_move(3)
    game.make_move(0)
    game.make_move(3)
    game.make_move(1)
    game.make_move(3)
    game.make_move(2)  # P1 has three vertical in col 3, now P1 to move
    assert game.get_winning_move() == 3


def test_get_blocking_move():
    """Blocking move returns column that blocks opponent's win."""
    game = ConnectFour()
    # P2 has three in a row (cols 1,2,3), P1 must block at 0 or 4
    game.board[5, 1] = -1
    game.board[5, 2] = -1
    game.board[5, 3] = -1
    game.current_player = 1  # P1 to move
    assert game.get_blocking_move() in (0, 4)

    game = ConnectFour()
    game.board[5, 3] = -1
    game.board[4, 3] = -1
    game.board[3, 3] = -1  # P2 has three vertical in col 3
    game.current_player = 1
    assert game.get_blocking_move() == 3


def test_get_winning_move_none():
    """No winning move returns None."""
    game = ConnectFour()
    assert game.get_winning_move() is None


def test_get_blocking_move_none():
    """No blocking needed returns None."""
    game = ConnectFour()
    assert game.get_blocking_move() is None


def test_winning_line():
    """Winning line returns correct positions."""
    game = ConnectFour()
    for col in [0, 1, 2, 3]:
        game.make_move(col)
        game.make_move(6)
    line = game.get_winning_line()
    assert line is not None
    assert len(line) == 4
    row = BOARD_ROWS - 1
    expected = [(row, 0), (row, 1), (row, 2), (row, 3)]
    assert set(line) == set(expected)
