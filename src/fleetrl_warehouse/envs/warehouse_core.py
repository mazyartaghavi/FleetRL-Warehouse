"""Dependency-light cooperative warehouse simulator.

The core simulator intentionally depends only on NumPy. Gymnasium wrappers live in
``gym_envs.py`` so tabular algorithms and unit tests remain easy to run.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from collections import deque
from typing import Iterable, Sequence

import numpy as np


class Action(IntEnum):
    """Discrete motion and manipulation actions."""

    WAIT = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    INTERACT = 5


@dataclass(slots=True)
class RobotState:
    robot_id: int
    position: tuple[int, int]
    carrying_task: int | None = None
    deliveries: int = 0


@dataclass(slots=True)
class Task:
    task_id: int
    pickup: tuple[int, int]
    delivery: tuple[int, int]
    status: str = "waiting"  # waiting, assigned, carried, delivered
    assigned_robot: int | None = None


@dataclass(slots=True)
class StepMetrics:
    reward: float
    collisions: int
    invalid_actions: int
    pickups: int
    deliveries: int
    completed: bool


class WarehouseCore:
    """Cooperative grid warehouse with simultaneous multi-robot actions.

    Robots share a team reward. A robot must move to a pickup station, execute
    ``INTERACT``, move to the corresponding delivery station, and execute
    ``INTERACT`` again. Simultaneous vertex collisions and edge swaps are blocked.
    """

    MOVES: dict[Action, tuple[int, int]] = {
        Action.WAIT: (0, 0),
        Action.NORTH: (0, -1),
        Action.SOUTH: (0, 1),
        Action.WEST: (-1, 0),
        Action.EAST: (1, 0),
        Action.INTERACT: (0, 0),
    }

    def __init__(
        self,
        width: int = 12,
        height: int = 10,
        n_robots: int = 3,
        n_tasks: int = 5,
        max_steps: int = 250,
        seed: int = 7,
        obstacles: Iterable[tuple[int, int]] | None = None,
    ) -> None:
        if width < 6 or height < 6:
            raise ValueError("Warehouse must be at least 6x6.")
        if n_robots < 1:
            raise ValueError("At least one robot is required.")
        if n_tasks < 1:
            raise ValueError("At least one task is required.")
        self.width = width
        self.height = height
        self.n_robots = n_robots
        self.n_tasks = n_tasks
        self.max_steps = max_steps
        self.base_seed = seed
        self.rng = np.random.default_rng(seed)
        self.obstacles = set(obstacles) if obstacles is not None else self._default_shelves()
        self.pickup_stations = self._default_pickups()
        self.delivery_stations = self._default_deliveries()
        self.robots: list[RobotState] = []
        self.tasks: list[Task] = []
        self.step_count = 0
        self.total_collisions = 0
        self.total_deliveries = 0
        self._last_metrics = StepMetrics(0.0, 0, 0, 0, 0, False)
        self.reset(seed=seed)

    def _default_shelves(self) -> set[tuple[int, int]]:
        shelves: set[tuple[int, int]] = set()
        for x in range(3, self.width - 2, 3):
            for y in range(2, self.height - 2):
                if y not in {self.height // 2}:
                    shelves.add((x, y))
        return shelves

    def _default_pickups(self) -> list[tuple[int, int]]:
        candidates = [(1, 1), (1, self.height - 2), (self.width // 2, 1)]
        return [p for p in candidates if self._inside(p) and p not in self.obstacles]

    def _default_deliveries(self) -> list[tuple[int, int]]:
        candidates = [
            (self.width - 2, 1),
            (self.width - 2, self.height - 2),
            (self.width // 2, self.height - 2),
        ]
        return [p for p in candidates if self._inside(p) and p not in self.obstacles]

    def _inside(self, p: tuple[int, int]) -> bool:
        return 0 <= p[0] < self.width and 0 <= p[1] < self.height

    def _free_cells(self) -> list[tuple[int, int]]:
        blocked = self.obstacles | set(self.pickup_stations) | set(self.delivery_stations)
        return [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in blocked
        ]

    def reset(self, seed: int | None = None) -> dict[str, object]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        free = self._free_cells()
        if len(free) < self.n_robots:
            raise RuntimeError("Not enough free cells for the requested robot fleet.")
        starts = self.rng.choice(len(free), size=self.n_robots, replace=False)
        self.robots = [
            RobotState(robot_id=i, position=free[int(index)])
            for i, index in enumerate(starts)
        ]
        self.tasks = []
        for task_id in range(self.n_tasks):
            pickup = self.pickup_stations[task_id % len(self.pickup_stations)]
            delivery = self.delivery_stations[(task_id + 1) % len(self.delivery_stations)]
            self.tasks.append(Task(task_id=task_id, pickup=pickup, delivery=delivery))
        self.step_count = 0
        self.total_collisions = 0
        self.total_deliveries = 0
        self._last_metrics = StepMetrics(0.0, 0, 0, 0, 0, False)
        return self.global_state()

    @staticmethod
    def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _target_for(self, robot: RobotState) -> tuple[int, int]:
        if robot.carrying_task is not None:
            return self.tasks[robot.carrying_task].delivery
        candidates = [t for t in self.tasks if t.status in {"waiting", "assigned"}]
        if not candidates:
            return robot.position
        task = min(candidates, key=lambda t: self.manhattan(robot.position, t.pickup))
        return task.pickup

    def _assign_waiting_tasks(self) -> None:
        """Greedy decentralized task assignment with one task per idle robot."""
        idle = [r for r in self.robots if r.carrying_task is None]
        waiting = [t for t in self.tasks if t.status == "waiting"]
        pairs: list[tuple[int, int, int]] = []
        for robot in idle:
            for task in waiting:
                pairs.append((self.manhattan(robot.position, task.pickup), robot.robot_id, task.task_id))
        used_robots: set[int] = set()
        used_tasks: set[int] = set()
        for _, robot_id, task_id in sorted(pairs):
            if robot_id in used_robots or task_id in used_tasks:
                continue
            self.tasks[task_id].status = "assigned"
            self.tasks[task_id].assigned_robot = robot_id
            used_robots.add(robot_id)
            used_tasks.add(task_id)

    def step(self, actions: Sequence[int | Action]) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
        if len(actions) != self.n_robots:
            raise ValueError(f"Expected {self.n_robots} actions, got {len(actions)}.")
        parsed = [Action(int(action)) for action in actions]
        self._assign_waiting_tasks()
        old_positions = [r.position for r in self.robots]
        old_targets = [self._target_for(r) for r in self.robots]
        old_distances = [self.manhattan(p, t) for p, t in zip(old_positions, old_targets, strict=True)]
        proposals = list(old_positions)
        invalid = 0

        for i, (robot, action) in enumerate(zip(self.robots, parsed, strict=True)):
            dx, dy = self.MOVES[action]
            candidate = (robot.position[0] + dx, robot.position[1] + dy)
            if action in {Action.INTERACT, Action.WAIT}:
                candidate = robot.position
            if not self._inside(candidate) or candidate in self.obstacles:
                candidate = robot.position
                invalid += 1
            proposals[i] = candidate

        collisions = 0
        blocked: set[int] = set()
        for i in range(self.n_robots):
            for j in range(i + 1, self.n_robots):
                same_target = proposals[i] == proposals[j] and proposals[i] != old_positions[i]
                edge_swap = proposals[i] == old_positions[j] and proposals[j] == old_positions[i]
                if same_target or edge_swap:
                    blocked.update({i, j})
                    collisions += 1
        for i in blocked:
            proposals[i] = old_positions[i]
        for robot, proposal in zip(self.robots, proposals, strict=True):
            robot.position = proposal

        movement_progress = sum(
            old_distance - self.manhattan(robot.position, target)
            for old_distance, robot, target in zip(old_distances, self.robots, old_targets, strict=True)
        )

        pickups = 0
        deliveries = 0
        for robot, action in zip(self.robots, parsed, strict=True):
            if action != Action.INTERACT:
                continue
            if robot.carrying_task is None:
                eligible = [
                    task
                    for task in self.tasks
                    if task.pickup == robot.position
                    and task.status in {"waiting", "assigned"}
                    and task.assigned_robot in {None, robot.robot_id}
                ]
                if eligible:
                    task = min(eligible, key=lambda t: t.task_id)
                    task.status = "carried"
                    task.assigned_robot = robot.robot_id
                    robot.carrying_task = task.task_id
                    pickups += 1
            else:
                task = self.tasks[robot.carrying_task]
                if robot.position == task.delivery:
                    task.status = "delivered"
                    robot.carrying_task = None
                    robot.deliveries += 1
                    deliveries += 1

        self.step_count += 1
        self.total_collisions += collisions
        self.total_deliveries += deliveries
        all_delivered = all(task.status == "delivered" for task in self.tasks)
        truncated = self.step_count >= self.max_steps and not all_delivered

        reward = -0.05
        reward += 0.08 * movement_progress
        reward -= 1.0 * collisions
        reward -= 0.20 * invalid
        reward += 1.0 * pickups
        reward += 10.0 * deliveries
        if deliveries > 1:
            reward += 2.0 * (deliveries - 1)
        if all_delivered:
            reward += 20.0

        self._last_metrics = StepMetrics(
            reward=reward,
            collisions=collisions,
            invalid_actions=invalid,
            pickups=pickups,
            deliveries=deliveries,
            completed=all_delivered,
        )
        info: dict[str, object] = {
            "collisions": collisions,
            "invalid_actions": invalid,
            "pickups": pickups,
            "deliveries": deliveries,
            "total_deliveries": self.total_deliveries,
            "completion_rate": self.total_deliveries / self.n_tasks,
            "step": self.step_count,
        }
        return self.global_state(), float(reward), all_delivered, truncated, info

    def local_state(self, robot_id: int) -> tuple[int, ...]:
        """Compact discrete state for tabular independent learners."""
        robot = self.robots[robot_id]
        target = self._target_for(robot)
        dx = int(np.clip(target[0] - robot.position[0], -3, 3))
        dy = int(np.clip(target[1] - robot.position[1], -3, 3))
        nearest_other = min(
            (
                (other.position[0] - robot.position[0], other.position[1] - robot.position[1])
                for other in self.robots
                if other.robot_id != robot_id
            ),
            key=lambda delta: abs(delta[0]) + abs(delta[1]),
            default=(0, 0),
        )
        odx = int(np.clip(nearest_other[0], -2, 2))
        ody = int(np.clip(nearest_other[1], -2, 2))
        return (
            robot.position[0],
            robot.position[1],
            int(robot.carrying_task is not None),
            dx,
            dy,
            odx,
            ody,
        )

    def global_vector(self) -> np.ndarray:
        """Normalized centralized observation used by deep-RL wrappers."""
        features: list[float] = []
        for robot in self.robots:
            target = self._target_for(robot)
            features.extend(
                [
                    robot.position[0] / max(1, self.width - 1),
                    robot.position[1] / max(1, self.height - 1),
                    float(robot.carrying_task is not None),
                    target[0] / max(1, self.width - 1),
                    target[1] / max(1, self.height - 1),
                ]
            )
        for task in self.tasks:
            status = {"waiting": 0.0, "assigned": 0.25, "carried": 0.5, "delivered": 1.0}[task.status]
            features.extend(
                [
                    task.pickup[0] / max(1, self.width - 1),
                    task.pickup[1] / max(1, self.height - 1),
                    task.delivery[0] / max(1, self.width - 1),
                    task.delivery[1] / max(1, self.height - 1),
                    status,
                ]
            )
        features.append(self.step_count / self.max_steps)
        return np.asarray(features, dtype=np.float32)

    def global_state(self) -> dict[str, object]:
        return {
            "robots": [
                {
                    "robot_id": r.robot_id,
                    "position": r.position,
                    "carrying_task": r.carrying_task,
                    "deliveries": r.deliveries,
                }
                for r in self.robots
            ],
            "tasks": [
                {
                    "task_id": t.task_id,
                    "pickup": t.pickup,
                    "delivery": t.delivery,
                    "status": t.status,
                    "assigned_robot": t.assigned_robot,
                }
                for t in self.tasks
            ],
            "step": self.step_count,
        }


    def shortest_path_action(self, robot_id: int) -> Action:
        """Return one obstacle-aware action toward the robot's current target.

        This is a deterministic planning baseline for demonstrations and algorithm
        comparison; it is deliberately separate from the learned policies.
        """
        robot = self.robots[robot_id]
        target = self._target_for(robot)
        if robot.position == target:
            return Action.INTERACT
        occupied = {r.position for r in self.robots if r.robot_id != robot_id}
        queue = deque([robot.position])
        parent: dict[tuple[int, int], tuple[int, int] | None] = {robot.position: None}
        action_to: dict[tuple[int, int], Action] = {}
        for_position = {
            Action.NORTH: (0, -1),
            Action.SOUTH: (0, 1),
            Action.WEST: (-1, 0),
            Action.EAST: (1, 0),
        }
        while queue:
            current = queue.popleft()
            if current == target:
                break
            for action, (dx, dy) in for_position.items():
                nxt = (current[0] + dx, current[1] + dy)
                if (
                    nxt in parent
                    or not self._inside(nxt)
                    or nxt in self.obstacles
                    or (nxt in occupied and nxt != target)
                ):
                    continue
                parent[nxt] = current
                action_to[nxt] = action
                queue.append(nxt)
        if target not in parent:
            return Action.WAIT
        cursor = target
        while parent[cursor] != robot.position:
            previous = parent[cursor]
            if previous is None:
                return Action.WAIT
            cursor = previous
        return action_to[cursor]

    def render_ansi(self) -> str:
        grid = [["." for _ in range(self.width)] for _ in range(self.height)]
        for x, y in self.obstacles:
            grid[y][x] = "#"
        for x, y in self.pickup_stations:
            grid[y][x] = "P"
        for x, y in self.delivery_stations:
            grid[y][x] = "D"
        for robot in self.robots:
            x, y = robot.position
            marker = str(robot.robot_id % 10)
            grid[y][x] = marker.lower() if robot.carrying_task is not None else marker
        legend = "Legend: # shelf, P pickup, D delivery, 0-9 robots (lowercase=carrying)"
        return "\n".join(" ".join(row) for row in grid) + "\n" + legend

    def render_rgb(self, cell_size: int = 24) -> np.ndarray:
        image = np.full((self.height * cell_size, self.width * cell_size, 3), 245, dtype=np.uint8)
        colors = {
            "shelf": np.array([70, 70, 70], dtype=np.uint8),
            "pickup": np.array([45, 160, 80], dtype=np.uint8),
            "delivery": np.array([220, 145, 35], dtype=np.uint8),
        }
        for x, y in self.obstacles:
            image[y * cell_size : (y + 1) * cell_size, x * cell_size : (x + 1) * cell_size] = colors["shelf"]
        for x, y in self.pickup_stations:
            image[y * cell_size : (y + 1) * cell_size, x * cell_size : (x + 1) * cell_size] = colors["pickup"]
        for x, y in self.delivery_stations:
            image[y * cell_size : (y + 1) * cell_size, x * cell_size : (x + 1) * cell_size] = colors["delivery"]
        palette = [
            np.array([50, 90, 210], dtype=np.uint8),
            np.array([175, 55, 175], dtype=np.uint8),
            np.array([30, 170, 180], dtype=np.uint8),
            np.array([205, 65, 75], dtype=np.uint8),
        ]
        for robot in self.robots:
            x, y = robot.position
            margin = max(2, cell_size // 6)
            image[
                y * cell_size + margin : (y + 1) * cell_size - margin,
                x * cell_size + margin : (x + 1) * cell_size - margin,
            ] = palette[robot.robot_id % len(palette)]
        return image
