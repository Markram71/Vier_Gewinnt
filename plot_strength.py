#!/usr/bin/env python3
"""
Plot playing-strength improvement over training iterations.

Two modes:
  1. From cached history (fast):
       python plot_strength.py
       Reads checkpoints/strength_history.json written by train_runner.py

  2. Scan all checkpoints (slower, but works without --strength-eval-interval):
       python plot_strength.py --scan
       Evaluates every checkpoint_iter_*.pt against the heuristic baseline.
       Results are cached so subsequent runs are instant.

Output: strength_over_training.png
"""

import argparse
import json
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe in all environments
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tqdm import tqdm

from src.config import get_device, CHECKPOINT_DIR
from src.network import ConnectFourNet
from src.strength_eval import eval_strength


def iter_from_path(p: Path) -> int:
    try:
        return int(p.stem.split("_")[-1])
    except (ValueError, IndexError):
        return -1


def load_network(path: Path, device: torch.device) -> ConnectFourNet:
    net = ConnectFourNet()
    state = torch.load(path, map_location=device, weights_only=False)
    net.load_state_dict(state.get("model", state))
    net.to(device)
    net.eval()
    return net


def scan_checkpoints(
    checkpoint_dir: Path,
    history_path: Path,
    n_games: int,
    device: torch.device,
) -> list:
    """Evaluate all checkpoint_iter_*.pt files; skip already-cached iterations."""
    # Load existing cache
    cached: dict[int, dict] = {}
    if history_path.exists():
        with open(history_path) as f:
            for entry in json.load(f).get("entries", []):
                cached[entry["iteration"]] = entry

    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_iter_*.pt"),
        key=iter_from_path,
    )
    if not checkpoints:
        print("No checkpoint_iter_*.pt files found in", checkpoint_dir)
        return []

    print(f"Found {len(checkpoints)} checkpoints. "
          f"{len(cached)} already cached, {len(checkpoints) - len(cached)} to evaluate.")

    new_entries = 0
    for ckpt_path in tqdm(checkpoints, desc="Evaluating checkpoints"):
        iteration = iter_from_path(ckpt_path)
        if iteration in cached:
            continue
        try:
            net = load_network(ckpt_path, device)
            strength = eval_strength(net, n_games=n_games, num_simulations=20, device=device)
            cached[iteration] = {
                "iteration": iteration,
                "vs_random": strength["vs_random"],
                "vs_heuristic": strength["vs_heuristic"],
                "n_games": n_games,
            }
            new_entries += 1
            # Save incrementally so progress is kept if the run is interrupted
            all_entries = sorted(cached.values(), key=lambda e: e["iteration"])
            with open(history_path, "w") as f:
                json.dump({"entries": all_entries}, f, indent=2)
        except Exception as e:
            tqdm.write(f"  Skipping {ckpt_path.name}: {e}")

    if new_entries > 0:
        print(f"Saved {len(cached)} entries to {history_path}")

    return sorted(cached.values(), key=lambda e: e["iteration"])


def plot(entries: list, output_path: Path) -> None:
    if not entries:
        print("No data to plot.")
        return

    iterations = [e["iteration"] for e in entries]
    vs_random = [e["vs_random"] * 100 for e in entries]
    vs_heuristic = [e["vs_heuristic"] * 100 for e in entries]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    ax.plot(iterations, vs_random, color="#f9a826", linewidth=2, label="vs Random agent", marker="o", markersize=4)
    ax.plot(iterations, vs_heuristic, color="#e94560", linewidth=2, label="vs Heuristic agent", marker="o", markersize=4)

    # Reference lines
    for level, style in ((50, "--"), (75, ":"), (90, ":")):
        ax.axhline(level, color="#444466", linewidth=0.8, linestyle=style)
        ax.text(iterations[0], level + 1.5, f"{level}%", color="#666688", fontsize=8)

    ax.set_xlim(min(iterations), max(iterations))
    ax.set_ylim(0, 105)
    ax.set_xlabel("Training iteration", color="#ccccdd", fontsize=11)
    ax.set_ylabel("Win rate (%)", color="#ccccdd", fontsize=11)
    ax.set_title("Connect Four AlphaZero — Playing strength over training", color="#eeeeff", fontsize=13)
    ax.tick_params(colors="#aaaacc")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334466")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d%%"))
    ax.legend(facecolor="#0f3460", edgecolor="#334466", labelcolor="#eeeeff", fontsize=10)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved chart to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Connect Four training strength improvement")
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan and evaluate all checkpoint_iter_*.pt files (cached on first run)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(CHECKPOINT_DIR),
        help=f"Checkpoint directory (default: {CHECKPOINT_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("strength_over_training.png"),
        help="Output image path (default: strength_over_training.png)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=20,
        help="Games per checkpoint when scanning (default: 20)",
    )
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu/cuda/mps)")
    args = parser.parse_args()

    history_path = args.checkpoint_dir / "strength_history.json"

    if args.scan:
        device = get_device(args.device)
        print(f"Device: {device}")
        entries = scan_checkpoints(args.checkpoint_dir, history_path, args.games, device)
    elif history_path.exists():
        with open(history_path) as f:
            entries = json.load(f).get("entries", [])
        print(f"Loaded {len(entries)} entries from {history_path}")
        if not entries:
            print("History file is empty. Run with --scan to evaluate checkpoints.")
            return
    else:
        print(
            f"No strength history found at {history_path}.\n"
            "Options:\n"
            "  • Run with --scan to evaluate all saved checkpoints\n"
            "  • Or train with --strength-eval-interval N to log during training"
        )
        return

    plot(entries, args.output)


if __name__ == "__main__":
    main()
