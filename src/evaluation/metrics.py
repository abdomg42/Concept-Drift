"""
Evaluation Framework
====================
Computes all drift-handling metrics:
- Detection delay
- False alarm rate
- Recovery time
- Pre/during/post-adaptation returns
- Catastrophic forgetting measure
- Transfer efficiency ratio
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EpisodeRecord:
    episode: int
    total_return: float
    drift_detected: bool = False
    drift_ground_truth: bool = False
    adaptation_active: bool = False


@dataclass
class ExperimentMetrics:
    # Returns by phase
    pre_drift_returns:    List[float] = field(default_factory=list)
    during_drift_returns: List[float] = field(default_factory=list)
    post_adapt_returns:   List[float] = field(default_factory=list)

    # Detection quality
    true_positives:  int = 0
    false_positives: int = 0
    false_negatives: int = 0
    detection_delays: List[int] = field(default_factory=list)

    # Recovery
    recovery_episodes: List[int] = field(default_factory=list)   # Episodes to 90% optimal

    # Forgetting
    old_env_returns_after_adaptation: List[float] = field(default_factory=list)

    # Timestamps
    true_drift_episodes:     List[int] = field(default_factory=list)
    detected_drift_episodes: List[int] = field(default_factory=list)

    def detection_rate(self) -> float:
        total = self.true_positives + self.false_negatives
        return self.true_positives / total if total > 0 else 0.0

    def false_alarm_rate(self) -> float:
        total = self.false_positives + self.true_positives
        return self.false_positives / total if total > 0 else 0.0

    def mean_detection_delay(self) -> float:
        return float(np.mean(self.detection_delays)) if self.detection_delays else float("inf")

    def mean_recovery_time(self) -> float:
        return float(np.mean(self.recovery_episodes)) if self.recovery_episodes else float("inf")

    def catastrophic_forgetting_score(self) -> float:
        """
        Returns how much performance dropped on the old environment
        after adapting to the new one. 0 = no forgetting.
        """
        if not self.pre_drift_returns or not self.old_env_returns_after_adaptation:
            return 0.0
        pre = np.mean(self.pre_drift_returns[-20:])
        post = np.mean(self.old_env_returns_after_adaptation)
        if abs(pre) < 1e-6:
            return 0.0
        return max(0.0, (pre - post) / (abs(pre) + 1e-8))

    def transfer_efficiency(self) -> float:
        """
        Performance gain per episode after adaptation starts.
        Higher = more sample-efficient adaptation.
        """
        if len(self.post_adapt_returns) < 2:
            return 0.0
        gain = np.mean(self.post_adapt_returns[-10:]) - self.post_adapt_returns[0]
        return gain / max(len(self.post_adapt_returns), 1)

    def summary(self) -> dict:
        return {
            "mean_pre_drift_return":      float(np.mean(self.pre_drift_returns))      if self.pre_drift_returns      else 0.0,
            "mean_during_drift_return":   float(np.mean(self.during_drift_returns))   if self.during_drift_returns   else 0.0,
            "mean_post_adapt_return":     float(np.mean(self.post_adapt_returns))     if self.post_adapt_returns     else 0.0,
            "detection_rate":             self.detection_rate(),
            "false_alarm_rate":           self.false_alarm_rate(),
            "mean_detection_delay":       self.mean_detection_delay(),
            "mean_recovery_time":         self.mean_recovery_time(),
            "catastrophic_forgetting":    self.catastrophic_forgetting_score(),
            "transfer_efficiency":        self.transfer_efficiency(),
        }


class MetricsTracker:
    """
    Stateful tracker updated each episode by the training loop.
    Handles phase transitions (pre-drift / during-drift / post-adapt).
    """

    def __init__(self, drift_episode: int, optimal_return: float, recovery_threshold: float = 0.9):
        self.drift_episode      = drift_episode
        self.optimal_return     = optimal_return
        self.recovery_threshold = recovery_threshold
        self.metrics            = ExperimentMetrics()
        self._detected_this_drift = False
        self._drift_detected_at   = -1
        self._adapting            = False
        self._recovery_start      = -1
        self._recovered           = False

    def update(self, episode: int, ep_return: float,
               drift_detected: bool, ground_truth_drift: bool):
        rec = EpisodeRecord(
            episode=episode,
            total_return=ep_return,
            drift_detected=drift_detected,
            drift_ground_truth=ground_truth_drift,
        )

        # Phase assignment
        if episode < self.drift_episode:
            self.metrics.pre_drift_returns.append(ep_return)
        elif not self._adapting:
            self.metrics.during_drift_returns.append(ep_return)
        else:
            self.metrics.post_adapt_returns.append(ep_return)

        # Ground truth drift book-keeping
        if ground_truth_drift and episode == self.drift_episode:
            self.metrics.true_drift_episodes.append(episode)
            self._detected_this_drift = False

        # Detection quality
        if drift_detected:
            self.metrics.detected_drift_episodes.append(episode)
            if ground_truth_drift or episode >= self.drift_episode:
                self.metrics.true_positives += 1
                if not self._detected_this_drift:
                    delay = max(0, episode - self.drift_episode)
                    self.metrics.detection_delays.append(delay)
                    self._detected_this_drift = True
                    self._drift_detected_at   = episode
                    self._adapting            = True
                    self._recovery_start      = episode
            else:
                self.metrics.false_positives += 1

        # Recovery tracking
        if self._adapting and not self._recovered:
            threshold = self.recovery_threshold * self.optimal_return
            if ep_return >= threshold:
                recovered_in = episode - self._recovery_start
                self.metrics.recovery_episodes.append(recovered_in)
                self._recovered = True

    def finalize(self) -> ExperimentMetrics:
        return self.metrics
