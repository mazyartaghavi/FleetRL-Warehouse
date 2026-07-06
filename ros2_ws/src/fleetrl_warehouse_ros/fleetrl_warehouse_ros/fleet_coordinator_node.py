"""Central warehouse task allocator and shared fleet-state publisher."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String


@dataclass
class Task:
    task_id: int
    pickup: tuple[float, float]
    delivery: tuple[float, float]
    status: str = "waiting"
    assigned_robot: int | None = None


class FleetCoordinator(Node):
    def __init__(self) -> None:
        super().__init__("fleet_coordinator")
        self.declare_parameter("robot_names", ["robot_0", "robot_1", "robot_2"])
        self.declare_parameter("goal_tolerance", 0.45)
        self.robot_names = list(self.get_parameter("robot_names").value)
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self.positions: dict[str, tuple[float, float]] = {}
        self.carrying: dict[str, int | None] = {name: None for name in self.robot_names}
        self.tasks = [
            Task(0, (-4.5, -3.0), (4.5, 3.0)),
            Task(1, (-4.5, 3.0), (4.5, -3.0)),
            Task(2, (0.0, -3.5), (4.5, 3.0)),
            Task(3, (-4.5, -3.0), (4.5, -3.0)),
            Task(4, (-4.5, 3.0), (4.5, 3.0)),
        ]
        self.state_pub = self.create_publisher(String, "/fleet/state", 10)
        self.interaction_sub = self.create_subscription(
            String, "/fleet/interactions", self._on_interaction, 10
        )
        for name in self.robot_names:
            self.create_subscription(
                Odometry,
                f"/{name}/odom",
                lambda msg, robot=name: self._on_odom(robot, msg),
                10,
            )
        self.create_timer(0.25, self._publish_state)
        self.get_logger().info(f"Managing {len(self.robot_names)} robots and {len(self.tasks)} tasks")

    def _on_odom(self, robot: str, msg: Odometry) -> None:
        self.positions[robot] = (msg.pose.pose.position.x, msg.pose.pose.position.y)

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _assign(self) -> None:
        idle = [name for name in self.robot_names if self.carrying[name] is None]
        waiting = [task for task in self.tasks if task.status == "waiting"]
        candidates: list[tuple[float, str, int]] = []
        for robot in idle:
            if robot not in self.positions:
                continue
            for task in waiting:
                candidates.append((self._distance(self.positions[robot], task.pickup), robot, task.task_id))
        used_robots: set[str] = set()
        used_tasks: set[int] = set()
        for _, robot, task_id in sorted(candidates):
            if robot in used_robots or task_id in used_tasks:
                continue
            task = self.tasks[task_id]
            task.status = "assigned"
            task.assigned_robot = self.robot_names.index(robot)
            used_robots.add(robot)
            used_tasks.add(task_id)

    def _on_interaction(self, msg: String) -> None:
        try:
            event = json.loads(msg.data)
            robot = str(event["robot"])
        except (json.JSONDecodeError, KeyError, TypeError):
            self.get_logger().warning("Ignoring malformed interaction event")
            return
        if robot not in self.positions:
            return
        position = self.positions[robot]
        carried = self.carrying[robot]
        if carried is None:
            assigned = [
                task
                for task in self.tasks
                if task.status == "assigned"
                and task.assigned_robot == self.robot_names.index(robot)
                and self._distance(position, task.pickup) <= self.goal_tolerance
            ]
            if assigned:
                task = assigned[0]
                task.status = "carried"
                self.carrying[robot] = task.task_id
                self.get_logger().info(f"{robot} picked task {task.task_id}")
        else:
            task = self.tasks[carried]
            if self._distance(position, task.delivery) <= self.goal_tolerance:
                task.status = "delivered"
                self.carrying[robot] = None
                self.get_logger().info(f"{robot} delivered task {task.task_id}")

    def _goal_for(self, robot: str) -> tuple[float, float]:
        carried = self.carrying[robot]
        if carried is not None:
            return self.tasks[carried].delivery
        assigned = [
            task
            for task in self.tasks
            if task.status == "assigned" and task.assigned_robot == self.robot_names.index(robot)
        ]
        if assigned:
            return assigned[0].pickup
        return self.positions.get(robot, (0.0, 0.0))

    def _publish_state(self) -> None:
        self._assign()
        payload = {
            "robots": [
                {
                    "name": name,
                    "robot_id": i,
                    "position": self.positions.get(name, (0.0, 0.0)),
                    "goal": self._goal_for(name),
                    "carrying_task": self.carrying[name],
                }
                for i, name in enumerate(self.robot_names)
            ],
            "tasks": [task.__dict__ for task in self.tasks],
            "delivered": sum(task.status == "delivered" for task in self.tasks),
        }
        msg = String()
        msg.data = json.dumps(payload)
        self.state_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FleetCoordinator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
