#!/usr/bin/env python3
"""CLI entry point for training the Connect Four AlphaZero network."""

import argparse
from collections import deque
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.config import (
    BATCH_SIZE,
    BUFFER_SIZE,
    CHECKPOINT_DIR,
    CHECKPOINT_EVERY,
    GAMES_PER_ITERATION,
    LEARNING_RATE,
    MCTS_SIMULATIONS,
    TRAINING_EPOCHS,
    VALUE_LOSS_WEIGHT,
    get_device,
)
from src.network import ConnectFourNet
from src.train import train_iteration


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Return path to checkpoint with highest iteration, or None if none exist."""
    checkpoints = list(checkpoint_dir.glob("checkpoint_iter_*.pt"))
    if not checkpoints:
        return None

    def iter_from_path(p: Path) -> int:
        try:
            return int(p.stem.split("_")[-1])
        except (ValueError, IndexError):
            return -1

    return max(checkpoints, key=iter_from_path)


def run_eval_gate(
    network: ConnectFourNet,
    best_model_path: Path,
    device: torch.device,
    num_games: int,
    num_simulations: int,
) -> bool:
    """
    Evaluate current network against best_model.pt.
    Returns True if current network wins ≥55% of games (should become new best).
    Skips if no best_model.pt exists yet (always saves on first time).
    """
    if not best_model_path.exists():
        return True

    from evaluate import evaluate
    stats = evaluate(
        model1_path=Path("__current__"),
        model2_path=best_model_path,
        num_games=num_games,
        num_simulations=num_simulations,
        device=device,
    )
    # evaluate() loads from file — we need a different approach for in-memory model
    # Save to a temp file, evaluate, then delete
    return stats["win_rate"] >= 0.55


def _eval_gate_with_temp(
    network: ConnectFourNet,
    best_model_path: Path,
    device: torch.device,
    num_games: int,
    num_simulations: int,
) -> bool:
    """Evaluate current in-memory network against saved best_model.pt."""
    if not best_model_path.exists():
        return True

    import tempfile
    from evaluate import evaluate

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = Path(f.name)

    try:
        torch.save({"model": network.state_dict()}, tmp_path)
        stats = evaluate(
            model1_path=tmp_path,
            model2_path=best_model_path,
            num_games=num_games,
            num_simulations=num_simulations,
            device=device,
        )
        win_rate = stats["win_rate"]
        tqdm.write(
            f"  Eval gate: {stats['wins']}W / {stats['draws']}D / {stats['losses']}L "
            f"({win_rate:.1%} win rate)"
        )
        return win_rate >= 0.55
    finally:
        tmp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Train Connect Four AlphaZero")
    parser.add_argument("--iterations", type=int, default=50, help="Number of training iterations")
    parser.add_argument("--quick", action="store_true", help="Quick test: 2 iters, 5 games, 20 sims")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, ignore existing checkpoints")
    parser.add_argument("--logdir", type=str, default="runs/connect4", help="TensorBoard log directory")
    parser.add_argument("--device", type=str, default=None, help="Device override (cpu/cuda/mps)")
    parser.add_argument(
        "--eval-gate",
        action="store_true",
        help="Only save best_model.pt if new model wins ≥55%% of eval games vs previous best",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=40,
        help="Games to play for eval gate (default: 40)",
    )
    args = parser.parse_args()

    # Quick mode: use smaller values without mutating the config module
    num_games = 5 if args.quick else GAMES_PER_ITERATION
    num_simulations = 20 if args.quick else MCTS_SIMULATIONS
    batch_size = 64 if args.quick else BATCH_SIZE
    training_epochs = 1 if args.quick else TRAINING_EPOCHS

    checkpoint_dir = Path(CHECKPOINT_DIR)
    resume_path = args.resume
    if resume_path is None and not args.no_resume:
        latest = find_latest_checkpoint(checkpoint_dir)
        if latest is not None:
            resume_path = str(latest)
            print(f"Auto-resuming from latest: {resume_path}")

    device = get_device(args.device)
    print(f"Using device: {device}")

    network = ConnectFourNet()
    if resume_path:
        state = torch.load(resume_path, map_location=device, weights_only=False)
        network.load_state_dict(state.get("model", state))
        print(f"Resumed from {resume_path}")
    network = network.to(device)

    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])

    buffer: deque = deque(maxlen=BUFFER_SIZE)
    writer = SummaryWriter(args.logdir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_iter = 0
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        start_iter = ckpt.get("iteration", 0)

    best_model_path = checkpoint_dir / "best_model.pt"

    for iteration in tqdm(range(start_iter, args.iterations), desc="Training"):
        metrics = train_iteration(
            network=network,
            buffer=buffer,
            optimizer=optimizer,
            device=device,
            iteration=iteration,
            writer=writer,
            num_games=num_games,
            num_simulations=num_simulations,
            training_epochs=training_epochs,
            batch_size=batch_size,
            value_loss_weight=VALUE_LOSS_WEIGHT,
        )

        if (iteration + 1) % 10 == 0 or iteration == 0:
            tqdm.write(
                f"Iter {iteration}: loss={metrics['loss']:.4f} "
                f"policy_loss={metrics['policy_loss']:.4f} "
                f"value_loss={metrics['value_loss']:.4f} "
                f"policy_acc={metrics['policy_acc']:.3f} "
                f"value_mse={metrics['value_mse']:.4f}"
            )

        if (iteration + 1) % CHECKPOINT_EVERY == 0:
            path = checkpoint_dir / f"checkpoint_iter_{iteration + 1}.pt"
            torch.save(
                {
                    "model": network.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iteration": iteration + 1,
                },
                path,
            )
            tqdm.write(f"Saved checkpoint to {path}")

            # Eval gate: only update best_model.pt if new model is demonstrably stronger
            if args.eval_gate:
                tqdm.write("Running eval gate...")
                is_better = _eval_gate_with_temp(
                    network, best_model_path, device, args.eval_games, num_simulations
                )
                if is_better:
                    torch.save({"model": network.state_dict()}, best_model_path)
                    tqdm.write(f"  New best model saved to {best_model_path}")
                else:
                    tqdm.write("  Current model did not beat previous best — keeping old best.")
            else:
                torch.save({"model": network.state_dict()}, best_model_path)

    # Final save without eval gate
    if not args.eval_gate:
        torch.save({"model": network.state_dict()}, best_model_path)
        tqdm.write(f"Saved best model to {best_model_path}")

    writer.close()


if __name__ == "__main__":
    main()
