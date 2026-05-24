#!/usr/bin/env python3
"""Generate a visualization of the Connect Four network architecture.

Usage:
    python scripts/visualize_network.py [--format png|pdf]

Requires: pip install torchviz

Output:
    connect4_net.png (or .pdf) in the project root
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.network import ConnectFourNet
from src.config import NUM_RES_BLOCKS, NUM_CHANNELS


def print_architecture():
    """Print text summary of the architecture."""
    params = sum(p.numel() for p in ConnectFourNet().parameters())
    print("Connect Four Network Architecture")
    print("=" * 40)
    print(f"Input:        (batch, 2, 6, 7) - board state")
    print(f"Conv stem:    2 -> {NUM_CHANNELS} channels, 3x3")
    print(f"Res blocks:   {NUM_RES_BLOCKS} x (Conv 3x3 -> BN -> ReLU -> Conv 3x3 -> BN -> Add residual -> ReLU)")
    print(f"Policy head:  Conv 1x1 -> BN -> ReLU -> Flatten -> Linear 84->7")
    print(f"Value head:   Conv 1x1 -> BN -> ReLU -> Flatten -> Linear 42->64 -> ReLU -> Linear 64->1 -> Tanh")
    print(f"Total params: {params:,}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Connect Four network")
    parser.add_argument(
        "--format",
        choices=["png", "pdf"],
        default="png",
        help="Output format (default: png)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "connect4_net",
        help="Output path without extension",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Print architecture summary only (no torchviz)",
    )
    args = parser.parse_args()

    if args.text_only:
        print_architecture()
        return

    try:
        from torchviz import make_dot
        from graphviz.backend.execute import ExecutableNotFound
    except ImportError as e:
        print("torchviz not installed. Showing text summary instead.")
        print("For graph viz: pip install torchviz")
        print()
        print_architecture()
        sys.exit(1)

    net = ConnectFourNet()
    x = torch.randn(1, 2, 6, 7)
    policy, value = net(x)
    y = policy.sum() + value.sum()
    dot = make_dot(y, params=dict(net.named_parameters()), show_attrs=False, show_saved=False)

    try:
        out_path = dot.render(str(args.output), format=args.format, cleanup=True)
        print(f"Saved: {out_path}")
    except ExecutableNotFound:
        print(
            "Graphviz is not installed or 'dot' is not on your PATH.\n"
            "Install the Graphviz system package, then run this script again:\n"
            "  macOS:  brew install graphviz\n"
            "  Ubuntu: sudo apt-get install graphviz\n"
            "  Windows: https://graphviz.org/download/",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
