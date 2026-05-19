"""
Drift-Controlled Environments
==============================
Wrappers around Gymnasium environments that inject configurable drift:
- CartPoleDrift:   sudden or gradual change in pole length / gravity
- MountainCarDrift: change in car force / gravity
- LunarLanderDrift: change in gravity

Each environment emits a `drift_active` flag for ground-truth evaluation.
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Any


class CartPoleDrift(gym.Wrapper):
    """
    CartPole-v1 with controllable drift in physics parameters.
    
    Drift modes:
      - 'sudden':   parameters jump instantly at drift_episode
      - 'gradual':  parameters linearly interpolate over drift_duration episodes
    
    Drift axes:
      - gravity:        default 9.8
      - masscart:       default 1.0
      - masspole:       default 0.1
      - length:         default 0.5
    """

    def __init__(
        self,
        drift_param: str = "gravity",
        drift_start: int = 150,
        drift_end_value: float = 14.0,
        drift_mode: str = "sudden",
        drift_duration: int = 50,
        seed: int = 42,
    ):
        base_env = gym.make("CartPole-v1")
        super().__init__(base_env)
        self.drift_param = drift_param
        self.drift_start = drift_start
        self.drift_end_value = drift_end_value
        self.drift_mode = drift_mode
        self.drift_duration = drift_duration
        self._episode = 0
        self.drift_active = False
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self._base_value = self._get_param()

    def _get_param(self) -> float:
        return getattr(self.env.unwrapped, self.drift_param)

    def _set_param(self, value: float):
        setattr(self.env.unwrapped, self.drift_param, value)

    def reset(self, **kwargs):
        self._episode += 1
        self._apply_drift()
        return self.env.reset(**kwargs)

    def _apply_drift(self):
        ep = self._episode
        if ep < self.drift_start:
            self._set_param(self._base_value)
            self.drift_active = False
        elif self.drift_mode == "sudden":
            self._set_param(self.drift_end_value)
            self.drift_active = True
        elif self.drift_mode == "gradual":
            if ep >= self.drift_start + self.drift_duration:
                self._set_param(self.drift_end_value)
            else:
                frac = (ep - self.drift_start) / self.drift_duration
                interp = self._base_value + frac * (self.drift_end_value - self._base_value)
                self._set_param(interp)
            self.drift_active = ep >= self.drift_start

    @property
    def current_param_value(self) -> float:
        return self._get_param()


class MountainCarDrift(gym.Wrapper):
    """
    MountainCar-v0 with drift in the force applied per step.
    
    A weaker force makes the task significantly harder.
    """
    def __init__(
        self,
        drift_start: int = 100,
        drift_end_value: float = 0.0005,   # Default is 0.001
        drift_mode: str = "sudden",
        seed: int = 42,
    ):
        base_env = gym.make("MountainCar-v0")
        super().__init__(base_env)
        self.drift_start = drift_start
        self.drift_end_value = drift_end_value
        self.drift_mode = drift_mode
        self._episode = 0
        self.drift_active = False
        self._base_power = self.env.unwrapped.power  # 0.001

    def reset(self, **kwargs):
        self._episode += 1
        if self._episode >= self.drift_start:
            if self.drift_mode == "sudden":
                self.env.unwrapped.power = self.drift_end_value
            self.drift_active = True
        else:
            self.env.unwrapped.power = self._base_power
            self.drift_active = False
        return self.env.reset(**kwargs)


class RewardDriftWrapper(gym.Wrapper):
    """
    Injects reward-structure drift: scales or negates the reward
    at a configurable episode, simulating a shifted objective.
    
    This tests whether the agent can detect and adapt to reward drift
    independent of environment dynamics changes.
    """
    def __init__(
        self,
        env: gym.Env,
        drift_start: int = 150,
        reward_scale_after: float = -0.5,  # Negate+scale reward after drift
    ):
        super().__init__(env)
        self._episode = 0
        self.drift_start = drift_start
        self.reward_scale_after = reward_scale_after
        self.drift_active = False

    def reset(self, **kwargs):
        self._episode += 1
        self.drift_active = self._episode >= self.drift_start
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.drift_active:
            reward = reward * self.reward_scale_after
        return obs, reward, terminated, truncated, info


class MultiDriftEnvironment:
    """
    Factory for creating a sequence of environments that simulate
    cross-environment transfer (different physics → different datasets).
    
    Returns environments one at a time; the caller switches to the next
    when a drift episode is reached.
    """

    ENVIRONMENTS = [
        ("CartPole-gravity",     "cartpole_gravity"),
        ("CartPole-length",      "cartpole_length"),
        ("CartPole-reward",      "cartpole_reward"),
        ("MountainCar-force",    "mountaincar_force"),
    ]

    @staticmethod
    def make(env_id: str, drift_start: int = 150, seed: int = 42) -> gym.Env:
        if env_id == "cartpole_gravity":
            return CartPoleDrift(
                drift_param="gravity",
                drift_start=drift_start,
                drift_end_value=14.0,
                seed=seed,
            )
        elif env_id == "cartpole_length":
            return CartPoleDrift(
                drift_param="length",
                drift_start=drift_start,
                drift_end_value=0.9,    # Longer pole → harder
                seed=seed,
            )
        elif env_id == "cartpole_reward":
            base = gym.make("CartPole-v1")
            return RewardDriftWrapper(base, drift_start=drift_start, reward_scale_after=-0.5)
        elif env_id == "mountaincar_force":
            return MountainCarDrift(drift_start=drift_start, seed=seed)
        else:
            return gym.make(env_id)
