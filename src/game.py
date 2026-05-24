"""Connect Four game logic and board representation."""

import numpy as np
from typing import List, Optional, Tuple

from .config import BOARD_ROWS, BOARD_COLS


class ConnectFour:
    """Connect Four game with 6 rows x 7 columns."""

    def __init__(self, board: Optional[np.ndarray] = None, current_player: int = 1):
        """Initialize game. Board is row-major: board[row][col]."""
        if board is not None:
            self.board = board.copy()
        else:
            self.board = np.zeros((BOARD_ROWS, BOARD_COLS), dtype=np.int8)
        self.current_player = current_player

    def get_valid_moves(self) -> List[int]:
        """Return list of valid column indices (0-6)."""
        return [c for c in range(BOARD_COLS) if self.board[0, c] == 0]

    def make_move(self, col: int) -> bool:
        """Make a move in the given column. Returns True if move was valid."""
        if col not in self.get_valid_moves():
            return False
        row = self._get_drop_row(col)
        self.board[row, col] = self.current_player
        self.current_player = -self.current_player
        return True

    def _get_drop_row(self, col: int) -> int:
        """Get the row where a stone would land in the given column."""
        for row in range(BOARD_ROWS - 1, -1, -1):
            if self.board[row, col] == 0:
                return row
        return -1

    def check_win(self) -> Optional[int]:
        """Return winner (1 or -1) if game over, else None."""
        for player in (1, -1):
            if self._has_four(player):
                return player
        return None

    def _has_four(self, player: int) -> bool:
        """Check if given player has four in a row."""
        # Horizontal
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS - 3):
                if all(self.board[row, col + i] == player for i in range(4)):
                    return True
        # Vertical
        for row in range(BOARD_ROWS - 3):
            for col in range(BOARD_COLS):
                if all(self.board[row + i, col] == player for i in range(4)):
                    return True
        # Diagonal (down-right)
        for row in range(BOARD_ROWS - 3):
            for col in range(BOARD_COLS - 3):
                if all(self.board[row + i, col + i] == player for i in range(4)):
                    return True
        # Diagonal (down-left)
        for row in range(BOARD_ROWS - 3):
            for col in range(3, BOARD_COLS):
                if all(self.board[row + i, col - i] == player for i in range(4)):
                    return True
        return False

    def get_winning_move(self) -> Optional[int]:
        """Return column that wins immediately for current player, or None."""
        for col in self.get_valid_moves():
            clone = self.clone()
            clone.make_move(col)
            if clone.check_win() == self.current_player:
                return col
        return None

    def get_blocking_move(self) -> Optional[int]:
        """Return column that blocks opponent's immediate win, or None."""
        opponent = -self.current_player
        for col in self.get_valid_moves():
            clone = self.clone()
            clone.current_player = opponent
            clone.make_move(col)
            if clone.check_win() == opponent:
                return col
        return None

    def get_winning_line(self) -> Optional[List[Tuple[int, int]]]:
        """Return list of (row, col) positions of winning four, or None."""
        for player in (1, -1):
            line = self._find_winning_line(player)
            if line:
                return line
        return None

    def _find_winning_line(self, player: int) -> Optional[List[Tuple[int, int]]]:
        """Find the four positions forming a win for player."""
        # Horizontal
        for row in range(BOARD_ROWS):
            for col in range(BOARD_COLS - 3):
                if all(self.board[row, col + i] == player for i in range(4)):
                    return [(row, col + i) for i in range(4)]
        # Vertical
        for row in range(BOARD_ROWS - 3):
            for col in range(BOARD_COLS):
                if all(self.board[row + i, col] == player for i in range(4)):
                    return [(row + i, col) for i in range(4)]
        # Diagonal (down-right)
        for row in range(BOARD_ROWS - 3):
            for col in range(BOARD_COLS - 3):
                if all(self.board[row + i, col + i] == player for i in range(4)):
                    return [(row + i, col + i) for i in range(4)]
        # Diagonal (down-left)
        for row in range(BOARD_ROWS - 3):
            for col in range(3, BOARD_COLS):
                if all(self.board[row + i, col - i] == player for i in range(4)):
                    return [(row + i, col - i) for i in range(4)]
        return None

    def is_draw(self) -> bool:
        """Return True if board is full with no winner."""
        return len(self.get_valid_moves()) == 0 and self.check_win() is None

    def is_terminal(self) -> bool:
        """Return True if game is over (win or draw)."""
        return self.check_win() is not None or self.is_draw()

    def clone(self) -> "ConnectFour":
        """Return a deep copy of the game state."""
        return ConnectFour(self.board.copy(), self.current_player)

    def to_tensor(self, canonical: bool = True) -> np.ndarray:
        """
        Convert board to 2-channel tensor for NN input.
        Channel 0: current player's stones (1)
        Channel 1: opponent's stones (-1)
        If canonical=True, always from current player's perspective.
        Shape: (2, BOARD_ROWS, BOARD_COLS)
        """
        p1 = (self.board == 1).astype(np.float32)
        p2 = (self.board == -1).astype(np.float32)
        if canonical and self.current_player == -1:
            p1, p2 = p2, p1
        return np.stack([p1, p2], axis=0)
