from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Optional, Tuple

from .replay_buffer import AdaptiveReplayBuffer


class Encoder(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, latent_dim), nn.ReLU(),
        )
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingQHead(nn.Module):
    def __init__(self, latent_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.value_stream = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        v = self.value_stream(z)
        a = self.advantage_stream(z)
        return v + (a - a.mean(dim=-1, keepdim=True))


class EnsembleDQN(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, latent_dim: int = 64, n_heads: int = 3):
        super().__init__()
        self.encoder = Encoder(obs_dim, latent_dim)
        self.heads = nn.ModuleList([
            DuelingQHead(latent_dim, n_actions) for _ in range(n_heads)
        ])
        self.n_heads = n_heads
        self.n_actions = n_actions

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        qs = torch.stack([head(z) for head in self.heads], dim=0)  # [K, B, A]
        return qs.mean(0), qs.std(0)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder(x)

    def q_values_all_heads(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return torch.stack([head(z) for head in self.heads], dim=0)


class DriftAwareDQN:

    def __init__(self, obs_dim: int, n_actions: int, config: dict, device: str = "cpu"):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = torch.device(device)
        cfg = config.get("agent", {})

        # Hyperparameters
        self.gamma           = cfg.get("gamma", 0.99)
        self.lr              = cfg.get("lr", 1e-3)
        self.batch_size      = cfg.get("batch_size", 64)
        self.tau             = cfg.get("tau", 0.005)          # Soft update rate
        self.eps_start       = cfg.get("eps_start", 1.0)
        self.eps_end         = cfg.get("eps_end", 0.05)
        self.eps_decay       = cfg.get("eps_decay", 0.995)
        self.eps_drift_boost = cfg.get("eps_drift_boost", 0.5)  # Boost on drift
        self.n_heads         = cfg.get("n_heads", 3)
        self.latent_dim      = cfg.get("latent_dim", 64)
        self.update_freq     = cfg.get("update_freq", 4)
        self._step_count     = 0

        # Networks
        self.online_net = EnsembleDQN(obs_dim, n_actions, self.latent_dim, self.n_heads).to(self.device)
        self.target_net = EnsembleDQN(obs_dim, n_actions, self.latent_dim, self.n_heads).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.lr)

        # Replay buffer
        buf_cfg = config.get("buffer", {})
        self.buffer = AdaptiveReplayBuffer(
            capacity=buf_cfg.get("capacity", 100_000),
            alpha=buf_cfg.get("alpha", 0.6),
            beta=buf_cfg.get("beta", 0.4),
            recency_decay=buf_cfg.get("recency_decay", 0.995),
        )

        self.epsilon = self.eps_start
        self.losses: list = []
        self.q_vars: list = []    # Ensemble disagreement trace

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(self, state: np.ndarray, greedy: bool = False) -> Tuple[int, np.ndarray, list]:
        """
        Select action via epsilon-greedy. Returns (action, embedding, q_estimates).
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.online_net.get_embedding(state_t).cpu().numpy().squeeze()
            qs_all = self.online_net.q_values_all_heads(state_t)  # [K, 1, A]
            qs_per_head = qs_all[:, 0, :].cpu().numpy()           # [K, A]
            q_estimates = qs_per_head.mean(axis=1).tolist()       # one Q per head
            mean_q, _ = self.online_net(state_t)
            action = int(mean_q.argmax(1).item())

        self.q_vars.append(float(np.var(q_estimates)))

        if not greedy and np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)

        return action, embedding, q_estimates

    # ------------------------------------------------------------------
    # Learning step
    # ------------------------------------------------------------------

    def learn(self) -> Optional[float]:
        if not self.buffer.ready:
            return None
        self._step_count += 1
        if self._step_count % self.update_freq != 0:
            return None

        states, actions, rewards, next_states, dones, indices, weights = \
            self.buffer.sample(self.batch_size)

        states_t      = torch.FloatTensor(states).to(self.device)
        actions_t     = torch.LongTensor(actions).to(self.device)
        rewards_t     = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t       = torch.FloatTensor(dones).to(self.device)
        weights_t     = torch.FloatTensor(weights).to(self.device)

        # Double DQN: online net selects action, target net evaluates
        with torch.no_grad():
            next_q_online, _ = self.online_net(next_states_t)
            next_actions = next_q_online.argmax(1)
            next_q_target, _ = self.target_net(next_states_t)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards_t + self.gamma * next_q * (1 - dones_t)

        # Current Q values (average ensemble for training stability)
        current_q, _ = self.online_net(states_t)
        current_q = current_q.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        td_errors = (targets - current_q).detach().cpu().numpy()
        loss = (weights_t * F.huber_loss(current_q, targets, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        # Update priorities and target network
        self.buffer.update_priorities(indices, td_errors)
        self._soft_update()
        self.losses.append(loss.item())

        return loss.item()

    def _soft_update(self):
        for param, target_param in zip(
            self.online_net.parameters(), self.target_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    # ------------------------------------------------------------------
    # Drift response
    # ------------------------------------------------------------------

    def on_drift_detected(self, discard_fraction: float = 0.5) -> None:
        """
        Called by the training loop when drift is confirmed.
        - Boosts epsilon for more exploration
        - Partially resets replay buffer
        - Reinitializes the top layers of online network (fine-tuning)
        """
        # Exploration boost
        self.epsilon = min(1.0, self.epsilon + self.eps_drift_boost)

        # Replay buffer: drop stale data
        self.buffer.on_drift_detected(discard_fraction=discard_fraction)

        # Fine-tune: reinitialize top layers while keeping encoder
        self._reinit_heads()

    def _reinit_heads(self):
        """Reinitialize Q-heads; keep encoder weights (transfer learning)."""
        for head in self.online_net.heads:
            for layer in head.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        # Sync target heads
        for t_head, o_head in zip(self.target_net.heads, self.online_net.heads):
            t_head.load_state_dict(o_head.state_dict())

    # ------------------------------------------------------------------
    # Epsilon management
    # ------------------------------------------------------------------

    def decay_epsilon(self):
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save({
            "online": self.online_net.state_dict(),
            "target": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online"])
        self.target_net.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]

    def store_transition(self, s, a, r, s2, done):
        self.buffer.push(s, a, r, s2, done)

    def increment_episode(self):
        self.buffer.increment_episode()
