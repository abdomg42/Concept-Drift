from __future__ import annotations
import numpy as np
from collections import namedtuple
from typing import Tuple

Transition = namedtuple(
    "Transition", ["state", "action", "reward", "next_state", "done", "timestamp"]
)


class AdaptiveReplayBuffer:

    def __init__(
        self,
        capacity: int = 100_000,
        alpha: float = 0.6,      # PER prioritization exponent
        beta: float = 0.4,       # IS correction exponent
        beta_annealing: float = 1e-4,
        recency_decay: float = 0.995,  # Weight decay per episode for old samples
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing = beta_annealing
        self.recency_decay = recency_decay

        self.buffer: list[Transition] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self._pos = 0
        self._size = 0
        self._max_priority = 1.0
        self._drift_timestamp: int = -1
        self._current_episode: int = 0

    def push(self, state, action, reward, next_state, done) -> None:
        t = Transition(
            np.array(state, dtype=np.float32),
            action,
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done),
            self._current_episode,
        )
        if self._size < self.capacity:
            self.buffer.append(t)
            self._size += 1
        else:
            self.buffer[self._pos] = t

        self.priorities[self._pos] = self._max_priority
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple:
        if self._size == 0:
            raise RuntimeError("Buffer is empty")

        raw_priorities = self.priorities[: self._size].copy()

        # Apply recency weighting after drift
        if self._drift_timestamp >= 0:
            age = np.array(
                [self._current_episode - t.timestamp for t in self.buffer[: self._size]],
                dtype=np.float32,
            )
            age_penalty = self.recency_decay ** age
            raw_priorities *= age_penalty

        probs = raw_priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self._size, batch_size, replace=False, p=probs)
        weights = (self._size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_annealing)

        batch = [self.buffer[i] for i in indices]
        states      = np.stack([t.state      for t in batch])
        actions     = np.array([t.action     for t in batch])
        rewards     = np.array([t.reward     for t in batch], dtype=np.float32)
        next_states = np.stack([t.next_state for t in batch])
        dones       = np.array([t.done       for t in batch], dtype=np.float32)

        return states, actions, rewards, next_states, dones, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        new_prios = (np.abs(td_errors) + 1e-6)
        self.priorities[indices] = new_prios
        self._max_priority = max(self._max_priority, new_prios.max())

    def on_drift_detected(self, discard_fraction: float = 0.5) -> None:
        """
        Called when drift is confirmed. Discards oldest `discard_fraction`
        of transitions to reduce stale data dominance.
        """
        self._drift_timestamp = self._current_episode
        n_keep = int(self._size * (1.0 - discard_fraction))
        # Keep the most recent transitions
        recent = sorted(self.buffer[: self._size], key=lambda t: t.timestamp)[-n_keep:]
        self.buffer = list(recent)
        self._size = len(self.buffer)
        self._pos = self._size % self.capacity
        # Reset priorities for kept transitions to max
        self.priorities[: self._size] = self._max_priority

    def increment_episode(self) -> None:
        self._current_episode += 1

    def __len__(self) -> int:
        return self._size

    @property
    def ready(self) -> bool:
        return self._size >= 64   # Minimum batch size
