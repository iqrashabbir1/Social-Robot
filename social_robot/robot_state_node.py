from __future__ import annotations

import math
from pathlib import Path

import psutil

from social_robot.config_helpers import nested_ros_params, resolve_ros_config
from social_robot.runtime import ensure_rclpy


class RobotStateNodeBase:
    def __init__(self) -> None:
        self._rclpy, Node = ensure_rclpy()

        class RobotStateNode(Node):
            def __init__(inner_self) -> None:
                super().__init__("robot_state_node")
                default_config = Path(__file__).resolve().parents[1] / "config" / "live_sensing.yaml"
                inner_self.declare_parameter("config_path", str(default_config))
                inner_self.declare_parameter("runtime_type", "")
                config = resolve_ros_config(inner_self.get_parameter("config_path").value, default_config)
                params = nested_ros_params(config, "robot_state_node")

                inner_self.pose_topic = str(params.get("pose_topic", "/robot_pose"))
                inner_self.health_topic = str(params.get("health_topic", "/system_health"))
                inner_self.publish_rate = float(params.get("publish_rate", 5.0))
                inner_self.frame_id = str(params.get("frame_id", "demo_base"))
                inner_self.placeholder_label = str(params.get("placeholder_label", "laptop_demo_context"))
                runtime_override = str(inner_self.get_parameter("runtime_type").value or "").strip()
                inner_self.runtime_type = str(runtime_override or params.get("runtime_type", "ros2_live_laptop_sensors"))
                inner_self.pose_pub = inner_self.create_publisher(__import__("geometry_msgs.msg", fromlist=["PoseStamped"]).PoseStamped, inner_self.pose_topic, 10)
                inner_self.health_pub = inner_self.create_publisher(__import__("diagnostic_msgs.msg", fromlist=["DiagnosticArray"]).DiagnosticArray, inner_self.health_topic, 10)
                inner_self.tick = 0
                inner_self.timer = inner_self.create_timer(max(0.1, 1.0 / max(inner_self.publish_rate, 0.1)), inner_self.publish_state)

            def publish_state(inner_self) -> None:
                PoseStamped = __import__("geometry_msgs.msg", fromlist=["PoseStamped"]).PoseStamped
                DiagnosticArray = __import__("diagnostic_msgs.msg", fromlist=["DiagnosticArray"]).DiagnosticArray
                DiagnosticStatus = __import__("diagnostic_msgs.msg", fromlist=["DiagnosticStatus"]).DiagnosticStatus
                KeyValue = __import__("diagnostic_msgs.msg", fromlist=["KeyValue"]).KeyValue

                stamp = inner_self.get_clock().now().to_msg()
                pose = PoseStamped()
                pose.header.stamp = stamp
                pose.header.frame_id = inner_self.frame_id
                pose.pose.position.x = math.sin(inner_self.tick / 10.0)
                pose.pose.position.y = math.cos(inner_self.tick / 10.0)
                pose.pose.position.z = 0.0
                pose.pose.orientation.w = 1.0
                inner_self.pose_pub.publish(pose)

                health = DiagnosticArray()
                health.header.stamp = stamp
                status = DiagnosticStatus()
                status.name = "social_robot/runtime"
                status.hardware_id = inner_self.placeholder_label
                status.level = DiagnosticStatus.OK
                status.message = "Laptop/demo placeholder state publisher"
                status.values = [
                    KeyValue(key="cpu_percent", value=f"{psutil.cpu_percent(interval=None):.2f}"),
                    KeyValue(key="memory_percent", value=f"{psutil.virtual_memory().percent:.2f}"),
                    KeyValue(key="runtime_label", value=inner_self.runtime_type),
                ]
                health.status = [status]
                inner_self.health_pub.publish(health)
                inner_self.tick += 1

        self.node_cls = RobotStateNode


def main() -> None:
    runtime = RobotStateNodeBase()
    rclpy = runtime._rclpy
    rclpy.init()
    node = runtime.node_cls()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
