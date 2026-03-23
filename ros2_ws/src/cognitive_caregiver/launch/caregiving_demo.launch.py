from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            Node(
                package="cognitive_caregiver",
                executable="risk_router_node",
                name="risk_router_node",
                output="screen",
            ),
            Node(
                package="cognitive_caregiver",
                executable="digital_twin_node",
                name="digital_twin_node",
                output="screen",
            ),
        ]
    )
