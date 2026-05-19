"""
Training Loop
=============
Integrates DriftAwareDQN + ConceptDriftDetectionSystem + AdaptationManager
across configurable environments with full metric tracking.
"""

from __future__ import annotations
import os
import sys
import random
import numpy as np
import torch
from tqdm import tqdm
from typing import Optional

# Add parent directory to path so src module can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.dqn_agent import DriftAwareDQN
from src.detection.drift_detectors import ConceptDriftDetectionSystem
from src.adaptation.strategies import AdaptationManager
from src.environments.drift_envs import MultiDriftEnvironment
from src.evaluation.metrics import MetricsTracker


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(config: dict, run_name: str = "run") -> dict:
    """
    Full training run. Returns a results dict with metrics and training curves.

    config keys:
      seed, n_episodes, drift_start, env_id, device, agent, buffer,
      drift_detection, adaptation, eval
    """
    set_seed(config.get("seed", 42))
    device = config.get("device", "cpu")

    # ------------------------------------------------------------------ #
    # Environment
    # ------------------------------------------------------------------ #
    env_id      = config.get("env_id", "cartpole_gravity")
    drift_start = config.get("drift_start", 150)
    env = MultiDriftEnvironment.make(env_id, drift_start=drift_start, seed=config.get("seed", 42))

    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # ------------------------------------------------------------------ #
    # Agent
    # ------------------------------------------------------------------ #
    agent = DriftAwareDQN(obs_dim, n_actions, config, device=device)

    # ------------------------------------------------------------------ #
    # Detection system
    # ------------------------------------------------------------------ #
    drift_detector = ConceptDriftDetectionSystem(config)

    # ------------------------------------------------------------------ #
    # Adaptation manager
    # ------------------------------------------------------------------ #
    adapter = AdaptationManager(agent, config)

    # ------------------------------------------------------------------ #
    # Metrics tracker
    # ------------------------------------------------------------------ #
    optimal_return = config.get("eval", {}).get("optimal_return", 500.0)
    tracker = MetricsTracker(
        drift_episode=drift_start,
        optimal_return=optimal_return,
    )

    # ------------------------------------------------------------------ #
    # Training curves (for plotting)
    # ------------------------------------------------------------------ #
    episode_returns:       list = []
    epsilon_trace:         list = []
    drift_flags:           list = []   # True on detected drift episode
    ground_truth_flags:    list = []
    loss_trace:            list = []
    q_variance_trace:      list = []

    n_episodes = config.get("n_episodes", 350)
    last_drift_detected_ep = -999
    recent_returns = []

    for episode in tqdm(range(1, n_episodes + 1), desc=run_name, ncols=80):
        obs, _ = env.reset()
        ep_return = 0.0
        ep_losses = []
        ep_reward  = 0.0

        done = False
        while not done:
            action, embedding, q_ests = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(obs, action, reward, next_obs, done)
            loss = agent.learn()
            if loss is not None:
                ep_losses.append(loss)

            ep_return += reward
            ep_reward  = reward     # Last reward used for statistical detectors
            obs = next_obs

        agent.decay_epsilon()
        agent.increment_episode()

        # Ground-truth drift flag from environment
        gt_drift = getattr(env, "drift_active", False)

        # Drift detection (per-episode call, using final reward for step-level detectors)
        drift_confirmed = drift_detector.step(
            episode_return=ep_return,
            reward=ep_reward,
            embedding=embedding,
            q_estimates=q_ests,
            episode=episode,
        )

        # Trigger adaptation if drift confirmed and not recently handled
        if drift_confirmed and (episode - last_drift_detected_ep) > 20:
            last_drift_detected_ep = episode
            agent.on_drift_detected(discard_fraction=0.4)
            # Build a tiny support batch for MAML if available
            support_batch = None
            if len(agent.buffer) >= 32:
                sb = agent.buffer.sample(32)
                support_batch = sb[:5]   # states, actions, rewards, next_states, dones
            adapter.on_drift(episode, support_batch=support_batch)
            drift_detector.reset_all()

        # Adaptation manager step
        recent_returns.append(ep_return)
        if len(recent_returns) > 20:
            recent_returns.pop(0)
        adapter.step(episode, float(np.mean(recent_returns)))

        # Metrics tracking
        tracker.update(episode, ep_return, drift_confirmed, gt_drift)

        # Trace logging
        episode_returns.append(ep_return)
        epsilon_trace.append(agent.epsilon)
        drift_flags.append(drift_confirmed)
        ground_truth_flags.append(gt_drift)
        loss_trace.append(float(np.mean(ep_losses)) if ep_losses else 0.0)
        q_var = float(np.mean(agent.q_vars[-50:])) if agent.q_vars else 0.0
        q_variance_trace.append(q_var)

    env.close()

    metrics = tracker.finalize()
    summary = metrics.summary()
    summary["run_name"] = run_name
    summary["env_id"]   = env_id

    return {
        "summary":             summary,
        "episode_returns":     episode_returns,
        "epsilon_trace":       epsilon_trace,
        "drift_flags":         drift_flags,
        "ground_truth_flags":  ground_truth_flags,
        "loss_trace":          loss_trace,
        "q_variance_trace":    q_variance_trace,
        "detection_timestamps": drift_detector.stats["detection_timestamps"],
        "metrics":             metrics,
    }


if __name__ == "__main__":
    import yaml
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "base.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Run training
    result = train(config, run_name="single_run")
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(result["summary"])
