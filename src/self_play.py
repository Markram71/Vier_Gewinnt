"""Self-play data generation for training."""

from typing import List, Tuple

import numpy as np

from .game import ConnectFour
from .mcts import MCTS
from .network import ConnectFourNet
from .config import BOARD_COLS, GAMES_PER_ITERATION, TEMPERATURE_THRESHOLD


def _synthetic_policy(col: int) -> np.ndarray:
    """One-hot policy with all mass on the given column."""
    policy = np.zeros(BOARD_COLS, dtype=np.float32)
    policy[col] = 1.0
    return policy


def generate_self_play_data(
    network: ConnectFourNet,
    num_games: int = GAMES_PER_ITERATION,
    num_simulations: int = 80,
    temperature_threshold: int = TEMPERATURE_THRESHOLD,
    device=None,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Generate training data by self-play.

    Returns:
        List of (state_tensor, policy_target, value_target)
        - state_tensor: (2, rows, cols) canonical from current player's view
        - policy_target: (7,) normalized visit counts
        - value_target: game outcome from current player's view (1, -1, or 0)
    """
    import torch
    device = device or torch.device("cpu")
    mcts = MCTS(network=network, num_simulations=num_simulations, device=device, add_noise=True)

    data: List[Tuple[np.ndarray, np.ndarray, float]] = []

    for _ in range(num_games):
        game = ConnectFour()
        game_data: List[Tuple[np.ndarray, np.ndarray]] = []

        while not game.is_terminal():
            move_number = int(np.sum(game.board != 0))
            temperature = 1.0 if move_number < temperature_threshold else 0.0
            state_tensor = game.to_tensor()

            # Check for forced moves: use synthetic targets instead of MCTS
            winning_col = game.get_winning_move()
            blocking_col = game.get_blocking_move()

            if winning_col is not None:
                # We can win: one-hot policy on winning move, play it
                policy_target = _synthetic_policy(winning_col)
                game_data.append((state_tensor.copy(), policy_target.copy()))
                game.make_move(winning_col)
            elif blocking_col is not None:
                # We must block: one-hot policy on blocking move, play it
                policy_target = _synthetic_policy(blocking_col)
                game_data.append((state_tensor.copy(), policy_target.copy()))
                game.make_move(blocking_col)
            else:
                # No forced move: run MCTS for policy target
                visit_counts = mcts.run(game)
                game_data.append((state_tensor.copy(), visit_counts.copy()))
                action = mcts.get_action(game, temperature=temperature)
                game.make_move(action)

        winner = game.check_win()
        if winner is not None:
            value_target = float(winner)
        else:
            value_target = 0.0

        for i, (state, policy) in enumerate(game_data):
            player = 1 if i % 2 == 0 else -1
            outcome = value_target * player
            data.append((state, policy, outcome))
            # Horizontal flip: board is left-right symmetric, doubling training data for free
            data.append((state[:, :, ::-1].copy(), policy[::-1].copy(), outcome))

    return data
