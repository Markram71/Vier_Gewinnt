"""Monte Carlo Tree Search for Connect Four."""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .game import ConnectFour
from .network import ConnectFourNet
from .config import BOARD_COLS, DIRICHLET_ALPHA, DIRICHLET_EPSILON, MCTS_SIMULATIONS, PUCT_C


class MCTSNode:
    """Node in the MCTS tree."""

    def __init__(
        self,
        state: ConnectFour,
        parent: Optional["MCTSNode"] = None,
        action: Optional[int] = None,
        prior: float = 0.0,
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children: Dict[int, MCTSNode] = {}
        self.visit_count = 0
        self.value_sum = 0.0

    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def ucb_score(self, c: float = PUCT_C) -> float:
        """Standard AlphaZero PUCT score: Q(s,a) + C·P(s,a)·√N(s) / (1 + N(s,a))."""
        parent_visits = self.parent.visit_count if self.parent else 1
        q = self.value_sum / self.visit_count if self.visit_count > 0 else 0.0
        u = c * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return q + u

    def select_child(self, c: float = PUCT_C) -> "MCTSNode":
        """Select child with highest UCB score."""
        return max(self.children.values(), key=lambda child: child.ucb_score(c))


class MCTS:
    """Monte Carlo Tree Search with neural network guidance."""

    def __init__(
        self,
        network: ConnectFourNet,
        num_simulations: int = MCTS_SIMULATIONS,
        device: Optional[torch.device] = None,
        add_noise: bool = False,
        dirichlet_alpha: float = DIRICHLET_ALPHA,
        dirichlet_epsilon: float = DIRICHLET_EPSILON,
    ):
        self.network = network
        self.num_simulations = num_simulations
        self.device = device or torch.device("cpu")
        self.add_noise = add_noise
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def run(self, state: ConnectFour) -> np.ndarray:
        """
        Run MCTS from the given state.

        Returns:
            visit_counts: (BOARD_COLS,) - normalized to get policy target
        """
        root = MCTSNode(state)
        # Expand root immediately so noise can be applied to child priors
        if not state.is_terminal():
            self._expand_and_backup(root, state.clone())
        if self.add_noise and root.is_expanded():
            actions = list(root.children.keys())
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
            eps = self.dirichlet_epsilon
            for i, a in enumerate(actions):
                root.children[a].prior = (1 - eps) * root.children[a].prior + eps * noise[i]

        for _ in range(self.num_simulations):
            node = root
            game = state.clone()

            # Selection
            while node.is_expanded() and not game.is_terminal():
                child_node = node.select_child()
                game.make_move(child_node.action)
                node = child_node

            # Expansion and backup
            if not game.is_terminal():
                value = self._expand_and_backup(node, game)
            else:
                winner = game.check_win()
                value = 1.0 if winner == 1 else (-1.0 if winner == -1 else 0.0)
                if game.current_player == -1:
                    value = -value

            # Backpropagation
            while node is not None:
                node.visit_count += 1
                node.value_sum += value
                value = -value
                node = node.parent

        visit_counts = np.zeros(BOARD_COLS, dtype=np.float32)
        for action, child in root.children.items():
            visit_counts[action] = child.visit_count
        total = visit_counts.sum()
        if total > 0:
            visit_counts /= total
        return visit_counts

    def _expand_and_backup(self, node: MCTSNode, game: ConnectFour) -> float:
        """Expand node with NN evaluation and return value from current player's view."""
        valid_moves = game.get_valid_moves()
        if not valid_moves:
            return 0.0

        board_tensor = game.to_tensor()
        board_batch = torch.from_numpy(board_tensor).float().unsqueeze(0).to(self.device)

        self.network.eval()
        with torch.no_grad():
            policy_logits, value = self.network(board_batch)

        policy = torch.softmax(policy_logits[0], dim=0).cpu().numpy()
        value = value.item()

        for action in valid_moves:
            prior = policy[action]
            child_state = game.clone()
            child_state.make_move(action)
            child_node = MCTSNode(child_state, parent=node, action=action, prior=prior)
            node.children[action] = child_node

        return value

    def get_action(
        self,
        state: ConnectFour,
        temperature: float = 0.0,
    ) -> int:
        """
        Get action from MCTS. temperature=0 for greedy, temperature=1 for exploration.
        Checks for winning/blocking moves first (don't rely on NN for obvious tactics).
        """
        valid_moves = state.get_valid_moves()
        if not valid_moves:
            raise ValueError("No valid moves")

        # Forced moves: always play win or block
        winning_col = state.get_winning_move()
        if winning_col is not None:
            return winning_col
        blocking_col = state.get_blocking_move()
        if blocking_col is not None:
            return blocking_col

        visit_counts = self.run(state)
        valid_counts = np.zeros(BOARD_COLS)
        for m in valid_moves:
            valid_counts[m] = visit_counts[m]
        valid_counts /= valid_counts.sum()

        if temperature <= 0:
            return int(np.argmax(valid_counts))
        else:
            probs = valid_counts ** (1.0 / temperature)
            probs /= probs.sum()
            return int(np.random.choice(BOARD_COLS, p=probs))
