"""Independent collision-safety layer that filters learned commands."""

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class SafetySupervisor(Node):
    def __init__(self) -> None:
        super().__init__("safety_supervisor")
        self.declare_parameter("robot_name", "robot_0")
        self.declare_parameter("stop_distance", 0.42)
        self.robot_name = str(self.get_parameter("robot_name").value)
        self.stop_distance = float(self.get_parameter("stop_distance").value)
        self.latest_raw = Twist()
        self.min_range = math.inf
        self.cmd_pub = self.create_publisher(Twist, f"/{self.robot_name}/cmd_vel", 10)
        self.create_subscription(Twist, f"/{self.robot_name}/cmd_vel_raw", self._on_cmd, 10)
        self.create_subscription(LaserScan, f"/{self.robot_name}/scan", self._on_scan, 10)
        self.create_timer(0.05, self._publish_safe)

    def _on_cmd(self, msg: Twist) -> None:
        self.latest_raw = msg

    def _on_scan(self, msg: LaserScan) -> None:
        valid = [r for r in msg.ranges if math.isfinite(r) and msg.range_min <= r <= msg.range_max]
        self.min_range = min(valid, default=math.inf)

    def _publish_safe(self) -> None:
        safe = Twist()
        safe.angular.z = self.latest_raw.angular.z
        if self.min_range > self.stop_distance or self.latest_raw.linear.x <= 0.0:
            safe.linear.x = self.latest_raw.linear.x
        else:
            safe.linear.x = 0.0
            safe.angular.z = max(0.6, abs(self.latest_raw.angular.z))
        self.cmd_pub.publish(safe)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SafetySupervisor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
