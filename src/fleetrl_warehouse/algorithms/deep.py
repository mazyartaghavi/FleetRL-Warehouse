"""Stable-Baselines3 training entry points for PPO, A2C, DDPG, and SAC."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def train_sb3(
    algorithm: str,
    total_timesteps: int,
    output: str | Path,
    env_config: dict[str, Any] | None = None,
    seed: int = 7,
):
    """Train a centralized policy and save the SB3 model.

    PPO/A2C use the multi-discrete environment. DDPG/SAC use the continuous
    velocity-like environment. Import errors include an actionable installation
    command so the lightweight tabular mode remains usable without PyTorch/SB3.
    """
    try:
        from stable_baselines3 import A2C, DDPG, PPO, SAC
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Deep-RL dependencies are missing. Run `pip install -e .[deep]`."
        ) from exc

    from fleetrl_warehouse.envs.gym_envs import WarehouseContinuousEnv, WarehouseDiscreteEnv

    name = algorithm.lower()
    if name in {"ppo", "a2c"}:
        env = Monitor(WarehouseDiscreteEnv(env_config))
        cls = {"ppo": PPO, "a2c": A2C}[name]
        kwargs = {"n_steps": 256} if name == "ppo" else {"n_steps": 32}
    elif name in {"ddpg", "sac"}:
        env = Monitor(WarehouseContinuousEnv(env_config))
        cls = {"ddpg": DDPG, "sac": SAC}[name]
        kwargs = {"buffer_size": 100_000, "learning_starts": 1_000}
    else:
        raise ValueError("algorithm must be one of: ppo, a2c, ddpg, sac")

    model = cls("MlpPolicy", env, verbose=1, seed=seed, **kwargs)
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output))
    env.close()
    return model
