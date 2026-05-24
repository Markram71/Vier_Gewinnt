"""Centralized hyperparameters for training and inference."""

# Board
BOARD_ROWS = 6
BOARD_COLS = 7

# Network — 5 blocks × 128 channels (~1.5M params), trains ~4× faster than the
# previous 10×192 setup while retaining full capacity for Connect Four
NUM_RES_BLOCKS = 5
NUM_CHANNELS = 128

# MCTS
MCTS_SIMULATIONS = 80
PUCT_C = 1.5
DIRICHLET_ALPHA = 0.3    # concentration param for root exploration noise
DIRICHLET_EPSILON = 0.25  # fraction of Dirichlet noise mixed into root priors

# Self-play
GAMES_PER_ITERATION = 150
TEMPERATURE_THRESHOLD = 10  # Use temp=1 for first N moves, then temp→0

# Training
BUFFER_SIZE = 20000
BATCH_SIZE = 256
TRAINING_EPOCHS = 5
LEARNING_RATE = 0.001
LR_DECAY_GAMMA = 0.99    # exponential LR decay per iteration (halves every ~70 iters)
VALUE_LOSS_WEIGHT = 1.0  # relative weight of value loss vs policy loss
WEIGHT_DECAY = 1e-4      # L2 regularisation in Adam
GRAD_CLIP = 1.0          # max gradient norm before clipping

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
