# Connect Four - AlphaZero

An AlphaZero-style implementation of Connect Four in Python with PyTorch. The neural network learns to play through self-play and Monte Carlo Tree Search (MCTS).

## Setup

```bash
pip install -r requirements.txt
```

## Training

Train the network with self-play:

```bash
python train_runner.py --iterations 50
```

Options:
- `--iterations N` - Number of training iterations (default: 50)
- `--resume PATH` - Resume from checkpoint
- `--logdir PATH` - TensorBoard log directory (default: runs/connect4)
- `--device cpu|cuda` - Device to use

Prevent Mac from sleeping during long training runs (built-in macOS utility):
```bash
caffeinate -i python train_runner.py --iterations 250
```
Use `-i` to prevent idle sleep. The process stays awake until training finishes or you stop it with Ctrl+C.

View training progress:
```bash
tensorboard --logdir runs/connect4
```

## Play Against the AI

### Web (browser)

1. Train a model (or use the untrained network)
2. Start the web server:
   ```bash
   python web/app.py
   ```
3. Open http://localhost:5000 in your browser

To reload the latest model (e.g. while training is running):
```bash
python restart_web.py
```
This stops the existing server and starts a new one, loading the most recent checkpoint or best_model.pt.

### Local (terminal)

Play in the terminal with a colored board:

```bash
python local/play.py
```

Options:
- `--checkpoint PATH` - Path to model checkpoint (default: checkpoints/best_model.pt)
- `--simulations N` - MCTS simulations per move (default: 50)
- `--device cpu|cuda` - Device for inference

You play as red; the AI plays as yellow.

## Network Visualization

Visualize the network architecture. You need **both** the Python package and the **Graphviz** system tools (the `dot` program):

```bash
pip install torchviz
# macOS: install Graphviz so `dot` is on your PATH
brew install graphviz
python scripts/visualize_network.py
```

Generates `connect4_net.png` in the project root. Use `--format pdf` for PDF. Use `--text-only` for a text summary without torchviz or Graphviz.

## Project Structure

```
Vier_Gewinnt/
├── src/
│   ├── game.py      # Connect Four rules
│   ├── network.py   # Policy + value neural network
│   ├── mcts.py      # Monte Carlo Tree Search
│   ├── self_play.py # Self-play data generation
│   ├── train.py     # Training loop
│   └── config.py    # Hyperparameters
├── web/
│   ├── app.py       # Flask server
│   └── templates/   # HTML frontend
├── local/
│   └── play.py      # Terminal game vs AI
├── checkpoints/     # Saved models
├── train_runner.py  # CLI for training
├── restart_web.py   # Restart web server (load latest model)
├── scripts/
│   └── visualize_network.py  # Generate network diagram
└── tests/           # Unit tests
```

## Tests

```bash
pytest tests/ -v
```
