from __future__ import annotations

import sys


def main() -> None:
    try:
        import cv2
        import rclpy
        from cv_bridge import CvBridge
        from rclpy.node import Node
        from sensor_msgs.msg import Image
    except Exception as exc:  # pragma: no cover - depends on Windows ROS install
        raise RuntimeError(
            "Windows camera publisher requires native ROS 2 Python, OpenCV, and cv_bridge on Windows."
        ) from exc

    class WindowsCameraPublisher(Node):
        def __init__(self) -> None:
            super().__init__("windows_camera_publisher")
            self.declare_parameter("camera_index", 0)
            self.declare_parameter("frame_rate", 10.0)
            self.declare_parameter("width", 640)
            self.declare_parameter("height", 480)
            self.declare_parameter("topic_name", "/camera/image_raw")

            self.camera_index = int(self.get_parameter("camera_index").value)
            self.frame_rate = float(self.get_parameter("frame_rate").value)
            self.width = int(self.get_parameter("width").value)
            self.height = int(self.get_parameter("height").value)
            self.topic_name = str(self.get_parameter("topic_name").value)

            self.bridge = CvBridge()
            self.publisher = self.create_publisher(Image, self.topic_name, 10)
            self.capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if self.capture.isOpened():
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            else:
                self.get_logger().warning(f"Webcam index {self.camera_index} could not be opened on Windows.")

            self.timer = self.create_timer(max(0.05, 1.0 / max(self.frame_rate, 0.1)), self.publish_frame)

        def publish_frame(self) -> None:
            if not self.capture.isOpened():
                return
            ok, frame = self.capture.read()
            if not ok or frame is None:
                self.get_logger().warning("Windows camera frame read failed.")
                return
            try:
                msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "windows_camera"
                self.publisher.publish(msg)
            except Exception as exc:
                self.get_logger().warning(f"Failed to publish Windows camera frame: {exc}")

        def destroy_node(self) -> bool:
            if hasattr(self, "capture") and self.capture is not None:
                self.capture.release()
            return super().destroy_node()

    rclpy.init(args=sys.argv)
    node = WindowsCameraPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
