from __future__ import annotations

import json
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class PhysiologyReplayNode(Node):
    def __init__(self) -> None:
        super().__init__("physiology_replay_node")
        self.publisher = self.create_publisher(String, "/caregiver/sensors/physiology", 10)
        self.data_path = Path("outputs/logs/physiology_ros2_bridge.jsonl")
        self.lines = self.data_path.read_text(encoding="utf-8").splitlines() if self.data_path.exists() else []
        self.index = 0
        self.timer = self.create_timer(0.75, self._publish_next)

    def _publish_next(self) -> None:
        if self.index >= len(self.lines):
            self.get_logger().info("Completed physiology replay.")
            self.timer.cancel()
            return
        payload = json.loads(self.lines[self.index])
        risk_score = min(
            1.0,
            max(
                0.0,
                0.18 * max((payload["heart_rate_bpm"] - 78.0) / 25.0, 0.0)
                + 0.24 * max((96.0 - payload["spo2_percent"]) / 4.0, 0.0)
                + 0.22 * max((payload["gait_variability"] - 0.28) / 0.22, 0.0),
            ),
        )
        payload["risk_score"] = round(risk_score, 4)
        msg = String()
        msg.data = json.dumps(payload)
        self.publisher.publish(msg)
        self.index += 1


def main() -> None:
    rclpy.init()
    node = PhysiologyReplayNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
