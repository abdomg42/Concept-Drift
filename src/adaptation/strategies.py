from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional


class FineTuner:

    def __init__(self, agent, config: dict):
        self.agent = agent
        cfg = config.get("adaptation", {})
        self.fine_tune_lr    = cfg.get("fine_tune_lr", 3e-4)
        self.thaw_episodes   = cfg.get("thaw_episodes", 20)
        self._drift_episode  = -1
        self._current_episode = 0

    def on_drift(self, episode: int):
        self._drift_episode = episode
        # Freeze encoder
        for p in self.agent.online_net.encoder.parameters():
            p.requires_grad = False
        # Higher LR for heads
        for g in self.agent.optimizer.param_groups:
            g["lr"] = self.fine_tune_lr

    def step(self, episode: int):
        self._current_episode = episode
        # Gradually unfreeze encoder after `thaw_episodes`
        if self._drift_episode > 0 and \
                (episode - self._drift_episode) >= self.thaw_episodes:
            for p in self.agent.online_net.encoder.parameters():
                p.requires_grad = True
            # Restore original LR
            for g in self.agent.optimizer.param_groups:
                g["lr"] = self.agent.lr
            self._drift_episode = -1   # Reset


class MAMLAdapter:

    def __init__(self, agent, config: dict):
        self.agent = agent
        cfg = config.get("adaptation", {})
        self.adapt_steps  = cfg.get("maml_adapt_steps", 10)
        self.adapt_lr     = cfg.get("maml_adapt_lr", 1e-3)
        self.meta_init    = None

    def save_meta_init(self):
        """Snapshot current weights as meta-initialization."""
        self.meta_init = copy.deepcopy(self.agent.online_net.state_dict())

    def adapt(self, support_batch: tuple):
        if self.meta_init is None:
            return  # No meta-init saved yet

        # Load meta-init
        self.agent.online_net.load_state_dict(self.meta_init)
        inner_opt = optim.Adam(
            self.agent.online_net.parameters(), lr=self.adapt_lr
        )

        states, actions, rewards, next_states, dones = support_batch
        states_t      = torch.FloatTensor(states)
        actions_t     = torch.LongTensor(actions)
        rewards_t     = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(next_states)
        dones_t       = torch.FloatTensor(dones)

        for _ in range(self.adapt_steps):
            with torch.no_grad():
                next_q, _ = self.agent.target_net(next_states_t)
                targets = rewards_t + self.agent.gamma * next_q.max(1)[0] * (1 - dones_t)

            q, _ = self.agent.online_net(states_t)
            q = q.gather(1, actions_t.unsqueeze(1)).squeeze(1)
            loss = nn.functional.huber_loss(q, targets)
            inner_opt.zero_grad()
            loss.backward()
            inner_opt.step()


class MultiPolicyEnsemble:
    """
    Maintains a pool of K policies (DQN snapshots).
    On each episode, selects the policy with highest estimated Q-value
    for the current state distribution, or blends them.
    
    Design: keep a circular buffer of policy checkpoints.
    After drift, the newest policy is initially random but quickly learns;
    older policies may perform better on the old concept.
    """

    def __init__(self, capacity: int = 5):
        self.capacity = capacity
        self.policies: list = []    # List of (state_dict, episode, mean_return)

    def add_policy(self, state_dict: dict, episode: int, mean_return: float):
        if len(self.policies) >= self.capacity:
            self.policies.pop(0)
        self.policies.append({
            "state": copy.deepcopy(state_dict),
            "episode": episode,
            "mean_return": mean_return,
        })

    def best_policy(self) -> Optional[dict]:
        """Return state dict of policy with highest stored return."""
        if not self.policies:
            return None
        return max(self.policies, key=lambda p: p["mean_return"])["state"]

    def latest_policy(self) -> Optional[dict]:
        if not self.policies:
            return None
        return self.policies[-1]["state"]

    def blend_q_values(self, net_class, obs: np.ndarray, weights: Optional[list] = None) -> np.ndarray:
        """Weighted average of Q-values across all stored policies."""
        if not self.policies:
            return np.zeros(1)
        if weights is None:
            weights = [1.0 / len(self.policies)] * len(self.policies)
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        q_agg = None
        for pol, w in zip(self.policies, weights):
            # This would need a reference network for forward pass
            # In practice, caller manages this
            pass
        return q_agg


class ProgressiveNetwork(nn.Module):
    """
    Progressive Neural Network: adds new 'columns' for new concepts
    while lateral connections allow reuse of prior features.
    
    Columns: list of Encoder + QHead pairs.
    When drift occurs, a new column is added; old columns are frozen.
    Lateral connections from old column activations are fed into new column.
    
    This is a simplified 2-column version illustrating the concept.
    """

    def __init__(self, obs_dim: int, n_actions: int, latent_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.latent_dim = latent_dim
        self.columns = nn.ModuleList()
        self.lateral_adapters = nn.ModuleList()
        self._add_column()   # Column 0

    def _add_column(self):
        n_existing = len(self.columns)
        col = nn.Sequential(
            nn.Linear(self.obs_dim + n_existing * self.latent_dim, 128), nn.ReLU(),
            nn.Linear(128, self.latent_dim), nn.ReLU(),
            nn.Linear(self.latent_dim, self.n_actions),
        )
        self.columns.append(col)

        if n_existing > 0:
            # Lateral adapter squeezes old latents to feed into new column
            adapter = nn.Linear(n_existing * self.latent_dim, n_existing * self.latent_dim)
            self.lateral_adapters.append(adapter)

    def expand(self):
        """Add new column for new concept. Freeze all old columns."""
        for col in self.columns:
            for p in col.parameters():
                p.requires_grad = False
        self._add_column()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.columns) == 1:
            return self.columns[0](x)

        # Collect outputs from all prior frozen columns (as lateral inputs)
        laterals = []
        with torch.no_grad():
            for col in self.columns[:-1]:
                # Extract intermediate latent (before final layer)
                z = col[:-1](x)    # All layers except final linear
                laterals.append(z)

        lateral_cat = torch.cat(laterals, dim=-1)
        aug_x = torch.cat([x, lateral_cat], dim=-1)
        return self.columns[-1](aug_x)


class AdaptationManager:
    """
    Orchestrates adaptation strategy selection and execution.
    Chooses between FineTuner, MAML, ensemble, or progressive
    based on configuration.
    """

    def __init__(self, agent, config: dict):
        self.agent = agent
        cfg = config.get("adaptation", {})
        self.strategy = cfg.get("strategy", "fine_tune")
        self.ensemble = MultiPolicyEnsemble(capacity=cfg.get("ensemble_size", 5))
        self._episode = 0

        if self.strategy in ("fine_tune", "both"):
            self.fine_tuner = FineTuner(agent, config)
        if self.strategy in ("maml", "both"):
            self.maml = MAMLAdapter(agent, config)

    def on_drift(self, episode: int, support_batch: Optional[tuple] = None):
        """Called when drift is confirmed."""
        print(f"  [Adaptation] Drift at episode {episode}, strategy={self.strategy}")
        if self.strategy == "fine_tune":
            self.fine_tuner.on_drift(episode)
        elif self.strategy == "maml" and support_batch is not None:
            self.maml.adapt(support_batch)
        elif self.strategy == "ensemble":
            # Switch to best-known policy
            best = self.ensemble.best_policy()
            if best:
                self.agent.online_net.load_state_dict(best)

    def step(self, episode: int, mean_return: float):
        """Called each episode for ongoing management."""
        self._episode = episode
        if self.strategy == "fine_tune":
            self.fine_tuner.step(episode)
        if self.strategy == "maml" and episode % 50 == 0:
            self.maml.save_meta_init()

        # Always snapshot for ensemble
        self.ensemble.add_policy(
            self.agent.online_net.state_dict(), episode, mean_return
        )
