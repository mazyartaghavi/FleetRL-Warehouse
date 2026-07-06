"""Evaluation helpers."""

from __future__ import annotations

from statistics import mean

from fleetrl_warehouse.algorithms.tabular import IndependentTabularFleet
from fleetrl_warehouse.envs.warehouse_core import WarehouseCore


def evaluate_tabular(
    agent: IndependentTabularFleet,
    env: WarehouseCore,
    episodes: int = 20,
    seed: int = 1000,
) -> dict[str, float]:
    rewards: list[float] = []
    completion: list[float] = []
    collisions: list[float] = []
    steps: list[float] = []
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    try:
        for episode in range(episodes):
            env.reset(seed=seed + episode)
            total_reward = 0.0
            episode_collisions = 0
            for _ in range(env.max_steps):
                states = [env.local_state(i) for i in range(env.n_robots)]
                actions = agent.act(states, explore=False)
                _, reward, terminated, truncated, info = env.step(actions)
                total_reward += reward
                episode_collisions += int(info["collisions"])
                if terminated or truncated:
                    break
            rewards.append(total_reward)
            completion.append(env.total_deliveries / env.n_tasks)
            collisions.append(float(episode_collisions))
            steps.append(float(env.step_count))
    finally:
        agent.epsilon = original_epsilon
    return {
        "mean_reward": mean(rewards),
        "mean_completion_rate": mean(completion),
        "mean_collisions": mean(collisions),
        "mean_steps": mean(steps),
    }
