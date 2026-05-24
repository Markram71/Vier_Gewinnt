"""Policy and value neural network for Connect Four."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .config import BOARD_ROWS, BOARD_COLS, NUM_RES_BLOCKS, NUM_CHANNELS


class ResidualBlock(nn.Module):
    """Residual block with two 3x3 convolutions."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ConnectFourNet(nn.Module):
    """Policy + value network for Connect Four (AlphaZero-style)."""

    def __init__(
        self,
        in_channels: int = 2,
        num_res_blocks: int = NUM_RES_BLOCKS,
        channels: int = NUM_CHANNELS,
        num_actions: int = BOARD_COLS,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_actions = num_actions

        self.conv_input = nn.Conv2d(in_channels, channels, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(channels)

        self.res_blocks = nn.ModuleList(
            [ResidualBlock(channels) for _ in range(num_res_blocks)]
        )

        self.policy_conv = nn.Conv2d(channels, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * BOARD_ROWS * BOARD_COLS, num_actions)

        self.value_conv = nn.Conv2d(channels, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(BOARD_ROWS * BOARD_COLS, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Board state (batch, in_channels, rows, cols)

        Returns:
            policy_logits: (batch, num_actions)
            value: (batch, 1) in [-1, 1]
        """
        x = F.relu(self.bn_input(self.conv_input(x)))

        for block in self.res_blocks:
            x = block(x)

        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy_logits = self.policy_fc(policy)

        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = torch.tanh(self.value_fc2(F.relu(self.value_fc1(value))))

        return policy_logits, value

    def predict(
        self,         board_tensor: np.ndarray, device: Optional[torch.device] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Single prediction for inference.

        Args:
            board_tensor: (2, rows, cols) or (batch, 2, rows, cols)
            device: torch device

        Returns:
            policy: (num_actions,) probabilities
            value: scalar in [-1, 1]
        """
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            if board_tensor.ndim == 3:
                x = torch.from_numpy(board_tensor).float().unsqueeze(0)
            else:
                x = torch.from_numpy(board_tensor).float()
            x = x.to(device)
            policy_logits, value = self.forward(x)
            policy = F.softmax(policy_logits, dim=1)
            if policy.size(0) == 1:
                return policy.cpu().numpy()[0], value.item()
            return policy.cpu().numpy(), value.cpu().numpy()
