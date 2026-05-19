"""
Concept Drift Detection Module
================================
Implements multiple drift detection strategies:
- Performance-based (rolling return monitoring)
- Statistical tests (KS test, Page-Hinkley, ADWIN)
- Representation-based (KL divergence in latent space)
- Ensemble disagreement
"""

from __future__ import annotations
import numpy as np
from collections import deque
from scipy import stats
from typing import Optional
import torch
import torch.nn.functional as F


class PerformanceMonitor:
    """
    Monitors rolling average of returns and triggers drift
    if performance drops exceed a configurable threshold.
    """
    def __init__(self, window: int = 50, threshold: float = 0.20, min_samples: int = 20):
        self.window = window
        self.threshold = threshold          # 20% drop triggers alarm
        self.min_samples = min_samples
        self.returns: deque = deque(maxlen=window * 2)
        self.baseline_mean: Optional[float] = None
        self.drift_detected = False

    def update(self, episode_return: float) -> bool:
        self.returns.append(episode_return)
        if len(self.returns) < self.min_samples:
            return False

        recent = list(self.returns)[-self.window:]
        recent_mean = np.mean(recent)

        # Establish baseline from first full window
        if self.baseline_mean is None and len(self.returns) >= self.window:
            self.baseline_mean = np.mean(list(self.returns)[:self.window])
            return False

        if self.baseline_mean is None:
            return False

        # Avoid division by zero / very small baselines
        if abs(self.baseline_mean) < 1e-6:
            return False

        drop = (self.baseline_mean - recent_mean) / (abs(self.baseline_mean) + 1e-8)
        self.drift_detected = drop > self.threshold
        return self.drift_detected

    def reset_baseline(self):
        self.baseline_mean = None
        self.returns.clear()
        self.drift_detected = False


class KSTestDetector:
    """
    Kolmogorov-Smirnov test comparing recent vs reference reward distributions.
    Drift is flagged when the two distributions differ significantly.
    """
    def __init__(self, window: int = 100, significance: float = 0.05, min_samples: int = 30):
        self.window = window
        self.significance = significance
        self.min_samples = min_samples
        self.reference: deque = deque(maxlen=window)
        self.recent: deque = deque(maxlen=window)
        self._reference_full = False

    def update(self, reward: float) -> bool:
        if not self._reference_full:
            self.reference.append(reward)
            if len(self.reference) >= self.window:
                self._reference_full = True
            return False

        self.recent.append(reward)
        if len(self.recent) < self.min_samples:
            return False

        stat, p_value = stats.ks_2samp(
            list(self.reference), list(self.recent)
        )
        return p_value < self.significance

    def reset(self):
        self.reference.clear()
        self.recent.clear()
        self._reference_full = False


class PageHinkleyDetector:
    """
    Page-Hinkley test: sequential change-point detection.
    Detects when the cumulative sum of deviations exceeds a threshold.
    """
    def __init__(self, delta: float = 0.005, lambda_: float = 50.0, alpha: float = 0.9999):
        self.delta = delta        # Minimum change to detect
        self.lambda_ = lambda_    # Detection threshold
        self.alpha = alpha        # Forgetting factor
        self._reset()

    def _reset(self):
        self.x_mean = 0.0
        self.m_t = 0.0            # Cumulative sum (increasing)
        self.U_t = 0.0            # Min so far
        self.n = 0

    def update(self, x: float) -> bool:
        self.n += 1
        self.x_mean = self.alpha * self.x_mean + (1 - self.alpha) * x
        self.m_t += x - self.x_mean - self.delta
        self.U_t = min(self.U_t, self.m_t)
        return (self.m_t - self.U_t) > self.lambda_

    def reset(self):
        self._reset()


class ADWINDetector:
    """
    ADWIN (Adaptive Windowing) algorithm for concept drift detection.
    Maintains a variable-size window and detects when sub-windows differ significantly.
    Simplified 1-D implementation suitable for scalar reward streams.
    """
    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self.window: list = []
        self.variance = 0.0
        self.total = 0.0

    def update(self, x: float) -> bool:
        self.window.append(x)
        n = len(self.window)
        if n < 2:
            return False

        # Slide through possible cut-points
        eps_cut = self._epsilon_cut(n)
        arr = np.array(self.window)
        mean_total = arr.mean()

        for i in range(1, n):
            w0 = arr[:i]
            w1 = arr[i:]
            diff = abs(w0.mean() - w1.mean())
            if diff >= eps_cut:
                # Drop older half to adapt
                self.window = list(arr[i:])
                return True
        return False

    def _epsilon_cut(self, n: int) -> float:
        m = 1.0 / (1.0 / max(n // 2, 1) + 1.0 / max(n - n // 2, 1))
        return np.sqrt((1.0 / (2 * m)) * np.log(4 * n / self.delta))

    def reset(self):
        self.window.clear()


class RepresentationDriftDetector:
    """
    Detects drift in the agent's latent representation space.
    Computes Maximum Mean Discrepancy (MMD) between reference and current embeddings.
    High MMD indicates the agent is seeing different kinds of states.
    """
    def __init__(self, window: int = 200, threshold: float = 0.1, bandwidth: float = 1.0):
        self.window = window
        self.threshold = threshold
        self.bandwidth = bandwidth
        self.reference_embeddings: list = []
        self.current_embeddings: list = []
        self._ref_locked = False

    def update(self, embedding: np.ndarray) -> bool:
        if not self._ref_locked:
            self.reference_embeddings.append(embedding)
            if len(self.reference_embeddings) >= self.window:
                self._ref_locked = True
            return False

        self.current_embeddings.append(embedding)
        if len(self.current_embeddings) >= self.window:
            mmd = self._compute_mmd(
                np.array(self.reference_embeddings[-self.window:]),
                np.array(self.current_embeddings[-self.window:])
            )
            self.current_embeddings = self.current_embeddings[-self.window // 2:]
            return mmd > self.threshold
        return False

    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray) -> float:
        XX = np.sum(X ** 2, axis=1, keepdims=True)
        YY = np.sum(Y ** 2, axis=1, keepdims=True)
        dist = XX + YY.T - 2 * X @ Y.T
        return np.exp(-dist / (2 * self.bandwidth ** 2)).mean()

    def _compute_mmd(self, X: np.ndarray, Y: np.ndarray) -> float:
        return self._rbf_kernel(X, X) + self._rbf_kernel(Y, Y) - 2 * self._rbf_kernel(X, Y)

    def reset(self):
        self.reference_embeddings.clear()
        self.current_embeddings.clear()
        self._ref_locked = False


class EnsembleDisagreementDetector:
    """
    High variance among ensemble value estimates signals potential drift.
    Works with any set of Q-value estimates.
    """
    def __init__(self, threshold: float = 2.0, window: int = 50):
        self.threshold = threshold
        self.window = window
        self.variance_history: deque = deque(maxlen=window)

    def update(self, q_estimates: list[float]) -> bool:
        if len(q_estimates) < 2:
            return False
        var = np.var(q_estimates)
        self.variance_history.append(var)
        if len(self.variance_history) < 10:
            return False
        baseline_var = np.mean(list(self.variance_history)[: len(self.variance_history) // 2])
        return var > baseline_var * self.threshold

    def reset(self):
        self.variance_history.clear()


class ConceptDriftDetectionSystem:
    """
    Unified interface combining all detection strategies with voting mechanism.
    Drift is confirmed when at least `quorum` detectors agree.
    
    Design philosophy: individual detectors are noisy; ensemble voting reduces
    false positives while maintaining sensitivity.
    """
    def __init__(self, config: dict):
        cfg = config.get("drift_detection", {})
        self.quorum = cfg.get("quorum", 2)           # Votes needed to confirm drift
        self.cooldown_episodes = cfg.get("cooldown", 30)
        self._cooldown_counter = 0
        self._total_drifts = 0
        self._false_alarms = 0

        # Instantiate detectors based on config
        self.detectors = {}
        if cfg.get("use_performance", True):
            self.detectors["performance"] = PerformanceMonitor(
                window=cfg.get("perf_window", 50),
                threshold=cfg.get("perf_threshold", 0.20),
            )
        if cfg.get("use_ks", True):
            self.detectors["ks_test"] = KSTestDetector(
                window=cfg.get("ks_window", 100),
                significance=cfg.get("ks_alpha", 0.05),
            )
        if cfg.get("use_page_hinkley", True):
            self.detectors["page_hinkley"] = PageHinkleyDetector(
                delta=cfg.get("ph_delta", 0.005),
                lambda_=cfg.get("ph_lambda", 50.0),
            )
        if cfg.get("use_adwin", True):
            self.detectors["adwin"] = ADWINDetector(
                delta=cfg.get("adwin_delta", 0.002)
            )
        if cfg.get("use_representation", False):
            self.detectors["representation"] = RepresentationDriftDetector(
                window=cfg.get("repr_window", 200),
                threshold=cfg.get("repr_threshold", 0.1),
            )
        if cfg.get("use_ensemble", False):
            self.detectors["ensemble"] = EnsembleDisagreementDetector(
                threshold=cfg.get("ens_threshold", 2.0)
            )

        self.votes_history: list = []
        self.detection_timestamps: list = []

    def step(
        self,
        episode_return: float,
        reward: float,
        embedding: Optional[np.ndarray] = None,
        q_estimates: Optional[list] = None,
        episode: int = 0,
    ) -> bool:
        """
        Update all detectors with latest observations.
        Returns True if drift is confirmed.
        """
        # Enforce cooldown to avoid repeated alarms for same drift event
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            # Still update detectors so they track current state
            self._update_all(episode_return, reward, embedding, q_estimates)
            return False

        self._update_all(episode_return, reward, embedding, q_estimates)
        votes = self._count_votes(episode_return, reward, embedding, q_estimates)
        self.votes_history.append(votes)

        if votes >= self.quorum:
            self._total_drifts += 1
            self._cooldown_counter = self.cooldown_episodes
            self.detection_timestamps.append(episode)
            return True
        return False

    def _update_all(self, episode_return, reward, embedding, q_estimates):
        if "performance" in self.detectors:
            self.detectors["performance"].update(episode_return)
        if "ks_test" in self.detectors:
            self.detectors["ks_test"].update(reward)
        if "page_hinkley" in self.detectors:
            self.detectors["page_hinkley"].update(reward)
        if "adwin" in self.detectors:
            self.detectors["adwin"].update(reward)
        if "representation" in self.detectors and embedding is not None:
            self.detectors["representation"].update(embedding)
        if "ensemble" in self.detectors and q_estimates is not None:
            self.detectors["ensemble"].update(q_estimates)

    def _count_votes(self, episode_return, reward, embedding, q_estimates) -> int:
        votes = 0
        if "performance" in self.detectors:
            votes += int(self.detectors["performance"].update(episode_return))
        if "ks_test" in self.detectors:
            votes += int(self.detectors["ks_test"].update(reward))
        if "page_hinkley" in self.detectors:
            votes += int(self.detectors["page_hinkley"].update(reward))
        if "adwin" in self.detectors:
            votes += int(self.detectors["adwin"].update(reward))
        if "representation" in self.detectors and embedding is not None:
            votes += int(self.detectors["representation"].update(embedding))
        if "ensemble" in self.detectors and q_estimates is not None:
            votes += int(self.detectors["ensemble"].update(q_estimates))
        return votes

    def reset_all(self):
        """Reset all detectors (called after confirmed adaptation)."""
        for det in self.detectors.values():
            det.reset() if hasattr(det, "reset") else None
        if "performance" in self.detectors:
            self.detectors["performance"].reset_baseline()

    @property
    def stats(self) -> dict:
        return {
            "total_drifts_detected": self._total_drifts,
            "detection_timestamps": self.detection_timestamps,
            "votes_history": self.votes_history,
        }
