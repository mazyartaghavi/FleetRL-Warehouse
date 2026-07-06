"""Convert a fleet policy action into raw differential-drive velocity commands."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String


class PolicyExecutor(Node):
    """Run a learned tabular policy when available, otherwise a deterministic baseline.

    The learned policy is trained in the fast grid simulator. Continuous Gazebo
    coordinates are discretized into grid cells before looking up a Q-table action.
    This separates high-level learning from low-level safety supervision.
    """

    def __init__(self) -> None:
        super().__init__("policy_executor")
        self.declare_parameter("robot_name", "robot_0")
        self.declare_parameter("robot_id", 0)
        self.declare_parameter("model_path", "")
        self.declare_parameter("linear_speed", 0.55)
        self.declare_parameter("angular_speed", 1.0)
        self.declare_parameter("goal_tolerance", 0.35)
        self.declare_parameter("cell_size", 1.0)
        self.declare_parameter("origin_x", -6.0)
        self.declare_parameter("origin_y", -5.0)
        self.robot_name = str(self.get_parameter("robot_name").value)
        self.robot_id = int(self.get_parameter("robot_id").value)
        self.linear_speed = float(self.get_parameter("linear_speed").value)
        self.angular_speed = float(self.get_parameter("angular_speed").value)
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self.cell_size = float(self.get_parameter("cell_size").value)
        self.origin_x = float(self.get_parameter("origin_x").value)
        self.origin_y = float(self.get_parameter("origin_y").value)
        self.position: tuple[float, float] | None = None
        self.yaw = 0.0
        self.fleet_state: dict | None = None
        self.q_table: dict[tuple[int, ...], np.ndarray] = {}
        self._load_model(str(self.get_parameter("model_path").value))
        self.cmd_pub = self.create_publisher(Twist, f"/{self.robot_name}/cmd_vel_raw", 10)
        self.interaction_pub = self.create_publisher(String, "/fleet/interactions", 10)
        self.create_subscription(Odometry, f"/{self.robot_name}/odom", self._on_odom, 10)
        self.create_subscription(String, "/fleet/state", self._on_state, 10)
        self.create_timer(0.10, self._control)

    def _load_model(self, model_path: str) -> None:
        if not model_path:
            self.get_logger().warning("No Q-table supplied; using the deterministic waypoint baseline")
            return
        path = Path(model_path).expanduser()
        if not path.exists():
            self.get_logger().warning(f"Model not found at {path}; using baseline")
            return
        payload = json.loads(path.read_text(encoding="utf-8"))
        table = payload["tables"][self.robot_id]
        self.q_table = {
            tuple(int(value) for value in key.split("|")): np.asarray(values, dtype=float)
            for key, values in table.items()
        }
        self.get_logger().info(f"Loaded {len(self.q_table)} states from {path}")

    def _on_odom(self, msg: Odometry) -> None:
        self.position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _on_state(self, msg: String) -> None:
        try:
            self.fleet_state = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning("Ignoring malformed fleet state")

    def _grid(self, p: tuple[float, float]) -> tuple[int, int]:
        return (
            int(round((p[0] - self.origin_x) / self.cell_size)),
            int(round((p[1] - self.origin_y) / self.cell_size)),
        )

    def _robot_record(self) -> dict | None:
        if not self.fleet_state:
            return None
        return next((r for r in self.fleet_state["robots"] if r["name"] == self.robot_name), None)

    def _tabular_state(self, record: dict) -> tuple[int, ...]:
        position = self._grid(tuple(record["position"]))
        goal = self._grid(tuple(record["goal"]))
        dx = int(np.clip(goal[0] - position[0], -3, 3))
        dy = int(np.clip(goal[1] - position[1], -3, 3))
        others = [r for r in self.fleet_state["robots"] if r["name"] != self.robot_name]
        if others:
            nearest = min(
                others,
                key=lambda r: math.hypot(r["position"][0] - record["position"][0], r["position"][1] - record["position"][1]),
            )
            other_grid = self._grid(tuple(nearest["position"]))
            odx = int(np.clip(other_grid[0] - position[0], -2, 2))
            ody = int(np.clip(other_grid[1] - position[1], -2, 2))
        else:
            odx = ody = 0
        return (position[0], position[1], int(record["carrying_task"] is not None), dx, dy, odx, ody)

    def _policy_action(self, record: dict) -> int | None:
        if not self.q_table:
            return None
        state = self._tabular_state(record)
        values = self.q_table.get(state)
        return int(np.argmax(values)) if values is not None else None

    def _publish_interaction(self) -> None:
        event = String()
        event.data = json.dumps({"robot": self.robot_name})
        self.interaction_pub.publish(event)

    def _go_to_goal(self, goal: tuple[float, float]) -> Twist:
        cmd = Twist()
        if self.position is None:
            return cmd
        dx = goal[0] - self.position[0]
        dy = goal[1] - self.position[1]
        distance = math.hypot(dx, dy)
        if distance <= self.goal_tolerance:
            self._publish_interaction()
            return cmd
        desired = math.atan2(dy, dx)
        error = math.atan2(math.sin(desired - self.yaw), math.cos(desired - self.yaw))
        cmd.angular.z = float(np.clip(1.8 * error, -self.angular_speed, self.angular_speed))
        if abs(error) < 0.45:
            cmd.linear.x = min(self.linear_speed, 0.6 * distance)
        return cmd

    def _control(self) -> None:
        record = self._robot_record()
        if record is None or self.position is None:
            return
        action = self._policy_action(record)
        goal = tuple(record["goal"])
        if action in {0, 5}:  # WAIT or INTERACT
            cmd = Twist()
            if action == 5 or math.hypot(goal[0] - self.position[0], goal[1] - self.position[1]) <= self.goal_tolerance:
                self._publish_interaction()
        else:
            # Learned high-level action influences a short local subgoal.
            offsets = {1: (0.0, -self.cell_size), 2: (0.0, self.cell_size), 3: (-self.cell_size, 0.0), 4: (self.cell_size, 0.0)}
            if action in offsets:
                ox, oy = offsets[action]
                local_goal = (self.position[0] + ox, self.position[1] + oy)
                cmd = self._go_to_goal(local_goal)
            else:
                cmd = self._go_to_goal(goal)
        if action is None:
            cmd = self._go_to_goal(goal)
        self.cmd_pub.publish(cmd)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PolicyExecutor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
