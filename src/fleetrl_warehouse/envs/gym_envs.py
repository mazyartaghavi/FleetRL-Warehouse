"""Gymnasium wrappers for centralized training / decentralized execution."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover - optional dependency guard
    raise ImportError(
        "Gymnasium is required for deep-RL environments. Install with `pip install -e .[deep]`."
    ) from exc

from .warehouse_core import Action, WarehouseCore


class WarehouseDiscreteEnv(gym.Env[np.ndarray, np.ndarray]):
    """Centralized multi-discrete environment for PPO and A2C.

    The policy outputs one discrete action per robot. A shared global observation
    and reward implement centralized training. The action components are applied
    independently to support decentralized execution at deployment time.
    """

    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 8}

    def __init__(self, config: dict[str, Any] | None = None, render_mode: str | None = None):
        cfg = config or {}
        self.core = WarehouseCore(**cfg)
        self.render_mode = render_mode
        self.action_space = spaces.MultiDiscrete([len(Action)] * self.core.n_robots)
        obs = self.core.global_vector()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=obs.shape, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.core.reset(seed=seed)
        return self.core.global_vector(), {"state": self.core.global_state()}

    def step(self, action: np.ndarray):
        _, reward, terminated, truncated, info = self.core.step(action.tolist())
        return self.core.global_vector(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            return self.core.render_ansi()
        if self.render_mode == "rgb_array":
            return self.core.render_rgb()
        return None


class WarehouseContinuousEnv(gym.Env[np.ndarray, np.ndarray]):
    """Continuous velocity-like action wrapper for DDPG and SAC.

    Each robot receives a high-level planar velocity ``[v_x, v_y]`` in [-1, 1].
    The grid simulator maps the dominant component to one safe motion primitive.
    Pickup/drop is automatic when a robot reaches its assigned target. ROS 2 then
    converts the resulting high-level motion into differential-drive commands.
    """

    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 8}

    def __init__(self, config: dict[str, Any] | None = None, render_mode: str | None = None):
        cfg = config or {}
        self.core = WarehouseCore(**cfg)
        self.render_mode = render_mode
        shape = (self.core.n_robots * 2,)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=shape, dtype=np.float32)
        obs = self.core.global_vector()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=obs.shape, dtype=np.float32)

    @staticmethod
    def _to_discrete(vx: float, vy: float) -> Action:
        if abs(vx) < 0.20 and abs(vy) < 0.20:
            return Action.WAIT
        if abs(vx) >= abs(vy):
            return Action.EAST if vx >= 0 else Action.WEST
        return Action.SOUTH if vy >= 0 else Action.NORTH

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.core.reset(seed=seed)
        return self.core.global_vector(), {"state": self.core.global_state()}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(self.core.n_robots, 2)
        discrete = []
        for robot, (vx, vy) in zip(self.core.robots, action, strict=True):
            if robot.position == self.core._target_for(robot):
                discrete.append(Action.INTERACT)
            else:
                discrete.append(self._to_discrete(float(vx), float(vy)))
        _, reward, terminated, truncated, info = self.core.step(discrete)
        return self.core.global_vector(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "ansi":
            return self.core.render_ansi()
        if self.render_mode == "rgb_array":
            return self.core.render_rgb()
        return None
