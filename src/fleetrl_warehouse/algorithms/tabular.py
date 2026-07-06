"""Independent Q-learning and SARSA baselines with shared team rewards."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import DefaultDict, Iterable

import numpy as np

from fleetrl_warehouse.envs.warehouse_core import Action, WarehouseCore

State = tuple[int, ...]


@dataclass(slots=True)
class TabularConfig:
    algorithm: str = "q_learning"
    learning_rate: float = 0.15
    gamma: float = 0.97
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    seed: int = 7

    def __post_init__(self) -> None:
        if self.algorithm not in {"q_learning", "sarsa"}:
            raise ValueError("algorithm must be 'q_learning' or 'sarsa'.")


class IndependentTabularFleet:
    """One Q-table per robot, trained from the shared cooperative reward."""

    def __init__(self, n_robots: int, config: TabularConfig | None = None) -> None:
        self.config = config or TabularConfig()
        self.n_robots = n_robots
        self.n_actions = len(Action)
        self.rng = np.random.default_rng(self.config.seed)
        self.epsilon = self.config.epsilon_start
        self.q_tables: list[DefaultDict[State, np.ndarray]] = [
            defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float64))
            for _ in range(n_robots)
        ]

    def act(self, states: Iterable[State], explore: bool = True) -> list[int]:
        actions: list[int] = []
        for robot_id, state in enumerate(states):
            if explore and self.rng.random() < self.epsilon:
                actions.append(int(self.rng.integers(self.n_actions)))
            else:
                values = self.q_tables[robot_id][state]
                best = np.flatnonzero(values == values.max())
                actions.append(int(self.rng.choice(best)))
        return actions

    def update_q_learning(
        self,
        states: list[State],
        actions: list[int],
        reward: float,
        next_states: list[State],
        done: bool,
    ) -> None:
        alpha = self.config.learning_rate
        gamma = self.config.gamma
        for i in range(self.n_robots):
            current = self.q_tables[i][states[i]][actions[i]]
            bootstrap = 0.0 if done else float(np.max(self.q_tables[i][next_states[i]]))
            target = reward + gamma * bootstrap
            self.q_tables[i][states[i]][actions[i]] = current + alpha * (target - current)

    def update_sarsa(
        self,
        states: list[State],
        actions: list[int],
        reward: float,
        next_states: list[State],
        next_actions: list[int],
        done: bool,
    ) -> None:
        alpha = self.config.learning_rate
        gamma = self.config.gamma
        for i in range(self.n_robots):
            current = self.q_tables[i][states[i]][actions[i]]
            bootstrap = 0.0 if done else self.q_tables[i][next_states[i]][next_actions[i]]
            target = reward + gamma * float(bootstrap)
            self.q_tables[i][states[i]][actions[i]] = current + alpha * (target - current)

    def train(self, env: WarehouseCore, episodes: int = 500) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        for episode in range(episodes):
            env.reset(seed=self.config.seed + episode)
            states = [env.local_state(i) for i in range(env.n_robots)]
            actions = self.act(states, explore=True)
            episode_reward = 0.0
            collisions = 0
            for _ in range(env.max_steps):
                _, reward, terminated, truncated, info = env.step(actions)
                done = terminated or truncated
                next_states = [env.local_state(i) for i in range(env.n_robots)]
                next_actions = self.act(next_states, explore=True)
                if self.config.algorithm == "q_learning":
                    self.update_q_learning(states, actions, reward, next_states, done)
                else:
                    self.update_sarsa(states, actions, reward, next_states, next_actions, done)
                episode_reward += reward
                collisions += int(info["collisions"])
                states, actions = next_states, next_actions
                if done:
                    break
            self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)
            history.append(
                {
                    "episode": float(episode),
                    "reward": float(episode_reward),
                    "completion_rate": float(env.total_deliveries / env.n_tasks),
                    "collisions": float(collisions),
                    "steps": float(env.step_count),
                    "epsilon": float(self.epsilon),
                }
            )
        return history

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "n_robots": self.n_robots,
            "epsilon": self.epsilon,
            "tables": [
                {"|".join(map(str, state)): values.tolist() for state, values in table.items()}
                for table in self.q_tables
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "IndependentTabularFleet":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        agent = cls(payload["n_robots"], TabularConfig(**payload["config"]))
        agent.epsilon = float(payload["epsilon"])
        for i, raw_table in enumerate(payload["tables"]):
            for key, values in raw_table.items():
                state = tuple(int(v) for v in key.split("|"))
                agent.q_tables[i][state] = np.asarray(values, dtype=np.float64)
        return agent
