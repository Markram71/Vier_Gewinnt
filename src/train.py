"""Training loop with progress visualization."""

import random
from collections import deque
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import (
    BATCH_SIZE,
    BOARD_COLS,
    BUFFER_SIZE,
    CHECKPOINT_DIR,
    CHECKPOINT_EVERY,
    GAMES_PER_ITERATION,
    LEARNING_RATE,
    MCTS_SIMULATIONS,
    TRAINING_EPOCHS,
    VALUE_LOSS_WEIGHT,
)
from .network import ConnectFourNet
from .self_play import generate_self_play_data


def train_iteration(
    network: ConnectFourNet,
    buffer: deque,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    iteration: int,
    writer: Optional[SummaryWriter] = None,
    num_games: int = GAMES_PER_ITERATION,
    num_simulations: int = MCTS_SIMULATIONS,
    training_epochs: int = TRAINING_EPOCHS,
    batch_size: int = BATCH_SIZE,
    value_loss_weight: float = VALUE_LOSS_WEIGHT,
) -> dict:
    """
    Run one training iteration: generate self-play data, add to buffer, train.

    Returns:
        Dict with loss, policy_loss, value_loss, policy_accuracy, value_mse
    """
    network.train()
    data = generate_self_play_data(
        network,
        num_games=num_games,
        num_simulations=num_simulations,
        device=device,
    )

    for item in data:
        buffer.append(item)

    if len(buffer) < batch_size:
        return {"loss": 0, "policy_loss": 0, "value_loss": 0, "policy_acc": 0, "value_mse": 0}

    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_policy_correct = 0
    total_value_sq_err = 0.0
    num_batches = 0

    for epoch in range(training_epochs):
        indices = list(range(len(buffer)))
        random.shuffle(indices)

        pbar = tqdm(
            range(0, len(indices), batch_size),
            desc=f"Iter {iteration} Epoch {epoch+1}/{training_epochs}",
            leave=False,
        )

        for start in pbar:
            batch_idx = indices[start : start + batch_size]
            if len(batch_idx) < batch_size // 2:
                break

            batch = [buffer[i] for i in batch_idx]
            states = torch.from_numpy(np.stack([b[0] for b in batch])).float().to(device)
            policy_targets = torch.from_numpy(np.stack([b[1] for b in batch])).float().to(device)
            value_targets = torch.from_numpy(np.array([b[2] for b in batch])).float().unsqueeze(1).to(device)

            optimizer.zero_grad()
            policy_logits, values = network(states)

            policy_loss = -(policy_targets * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
            value_loss = F.mse_loss(values, value_targets)
            loss = policy_loss + value_loss_weight * value_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            pred_policy = torch.argmax(policy_logits, dim=1)
            target_policy = torch.argmax(policy_targets, dim=1)
            total_policy_correct += (pred_policy == target_policy).sum().item()
            total_value_sq_err += ((values - value_targets) ** 2).sum().item()
            num_batches += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    n = num_batches * batch_size if num_batches else 1
    metrics = {
        "loss": total_loss / num_batches if num_batches else 0,
        "policy_loss": total_policy_loss / num_batches if num_batches else 0,
        "value_loss": total_value_loss / num_batches if num_batches else 0,
        "policy_acc": total_policy_correct / n if n else 0,
        "value_mse": total_value_sq_err / n if n else 0,
    }

    if writer:
        writer.add_scalar("train/loss", metrics["loss"], iteration)
        writer.add_scalar("train/policy_loss", metrics["policy_loss"], iteration)
        writer.add_scalar("train/value_loss", metrics["value_loss"], iteration)
        writer.add_scalar("train/policy_accuracy", metrics["policy_acc"], iteration)
        writer.add_scalar("train/value_mse", metrics["value_mse"], iteration)

    return metrics
