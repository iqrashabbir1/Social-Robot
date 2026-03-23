from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class RiskRouterNode(Node):
    def __init__(self) -> None:
        super().__init__("risk_router_node")
        self.subscription = self.create_subscription(String, "/caregiver/sensors/physiology", self._on_sensor_state, 10)
        self.publisher = self.create_publisher(String, "/caregiver/alerts/risk", 10)

    def _on_sensor_state(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning("Received malformed physiology JSON")
            return

        risk_score = float(payload.get("risk_score", 0.0))
        risk_level = "high" if risk_score >= 0.62 else "moderate" if risk_score >= 0.42 else "low"
        alert = String()
        alert.data = json.dumps({"risk_score": risk_score, "risk_level": risk_level})
        self.publisher.publish(alert)


def main() -> None:
    rclpy.init()
    node = RiskRouterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
