"""Flask web app for playing Connect Four against the trained AI."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import BOARD_ROWS, BOARD_COLS, get_device
from src.game import ConnectFour
from src.mcts import MCTS
from src.network import ConnectFourNet

app = Flask(__name__, template_folder="templates", static_folder="static")

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
network: Optional[ConnectFourNet] = None
mcts: Optional[MCTS] = None
_device: Optional[torch.device] = None


def get_latest_model_path() -> Optional[Path]:
    """Return path to most recently modified model (best_model.pt or checkpoint_iter_*.pt)."""
    best = CHECKPOINT_DIR / "best_model.pt"
    checkpoints = list(CHECKPOINT_DIR.glob("checkpoint_iter_*.pt"))
    candidates = ([best] if best.exists() else []) + checkpoints
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_model():
    global network, mcts, _device
    _device = get_device()
    model_path = get_latest_model_path()
    network = ConnectFourNet()
    if model_path is not None:
        try:
            state = torch.load(model_path, map_location=_device, weights_only=False)
            network.load_state_dict(state.get("model", state))
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}. Using untrained network.")
    else:
        print("No checkpoint found, using untrained network")
    network.to(_device)
    network.eval()
    mcts = MCTS(network=network, num_simulations=50, device=_device)


def validate_board(board_flat: list) -> Optional[str]:
    """Return an error string if the board is invalid, else None."""
    if len(board_flat) != BOARD_ROWS * BOARD_COLS:
        return f"Board must have {BOARD_ROWS * BOARD_COLS} cells, got {len(board_flat)}"
    valid = {-1, 0, 1}
    for v in board_flat:
        if v not in valid:
            return f"Invalid cell value {v!r}: must be -1, 0, or 1"
    ones = sum(1 for v in board_flat if v == 1)
    neg_ones = sum(1 for v in board_flat if v == -1)
    if abs(ones - neg_ones) > 1:
        return f"Inconsistent stone counts: {ones} vs {neg_ones}"
    return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/move", methods=["POST"])
def get_move():
    """Receive board state and current player, return AI move."""
    data = request.get_json()
    if not data or "board" not in data:
        return jsonify({"error": "Missing board"}), 400

    error = validate_board(data["board"])
    if error:
        return jsonify({"error": error}), 400

    current_player = data.get("currentPlayer", 1)
    if current_player not in (1, -1):
        return jsonify({"error": "currentPlayer must be 1 or -1"}), 400

    board = np.array(data["board"], dtype=np.int8).reshape(BOARD_ROWS, BOARD_COLS)
    game = ConnectFour(board=board, current_player=current_player)

    if game.is_terminal():
        winner = game.check_win()
        return jsonify({"action": None, "gameOver": True, "winner": int(winner) if winner else 0})

    if not game.get_valid_moves():
        return jsonify({"action": None, "gameOver": True, "winner": 0})

    action = mcts.get_action(game, temperature=0)
    return jsonify({"action": int(action), "gameOver": False, "winner": None})


@app.route("/check_win", methods=["POST"])
def check_win():
    """Check if the given board has a winner or is a draw."""
    data = request.get_json()
    if not data or "board" not in data:
        return jsonify({"error": "Missing board"}), 400

    error = validate_board(data["board"])
    if error:
        return jsonify({"error": error}), 400

    board = np.array(data["board"], dtype=np.int8).reshape(BOARD_ROWS, BOARD_COLS)
    game = ConnectFour(board=board)
    winner = game.check_win()
    line = game.get_winning_line()

    return jsonify({
        "winner": int(winner) if winner else 0,
        "winningLine": line if line else None,
        "isDraw": game.is_draw(),
    })


if __name__ == "__main__":
    load_model()
    app.run(debug=True, port=5000)
