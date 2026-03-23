from __future__ import annotations

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class DigitalTwinNode(Node):
    def __init__(self) -> None:
        super().__init__("digital_twin_node")
        self.publisher = self.create_publisher(String, "/caregiver/digital_twin/state", 10)
        self.subscription = self.create_subscription(String, "/caregiver/alerts/risk", self._on_risk_alert, 10)
        self.current_state = {
            "patient_id": "sim_patient",
            "risk_level": "low",
            "adherence_state": "unknown",
            "emotion_state": "unknown",
            "oversight_required": False,
        }

    def _on_risk_alert(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().warning("Received malformed JSON risk alert")
            return

        self.current_state["risk_level"] = payload.get("risk_level", self.current_state["risk_level"])
        self.current_state["oversight_required"] = self.current_state["risk_level"] in {"high", "critical"}
        out = String()
        out.data = json.dumps(self.current_state)
        self.publisher.publish(out)


def main() -> None:
    rclpy.init()
    node = DigitalTwinNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
