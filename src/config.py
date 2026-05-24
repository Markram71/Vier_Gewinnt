"""Centralized hyperparameters for training and inference."""

# Board
BOARD_ROWS = 6
BOARD_COLS = 7

# Network (larger for better playing strength)
NUM_RES_BLOCKS = 10
NUM_CHANNELS = 192

# MCTS
MCTS_SIMULATIONS = 80
PUCT_C = 1.5

# Self-play
GAMES_PER_ITERATION = 150
TEMPERATURE_THRESHOLD = 10  # Use temp=1 for first N moves, then temp→0

# Training
BUFFER_SIZE = 20000
BATCH_SIZE = 256
TRAINING_EPOCHS = 5
LEARNING_RATE = 0.001
VALUE_LOSS_WEIGHT = 1.0  # relative weight of value loss vs policy loss

# Checkpointing
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_EVERY = 5  # Save every N iterations


def get_device(override: str | None = None):
    """Return best available torch.device, or the specified override."""
    import torch
    if override:
        return torch.device(override)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
