from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    default_config = PathJoinSubstitution([FindPackageShare("social_robot"), "config", "playback_grounded.yaml"])
    config_path = LaunchConfiguration("config_path")
    output_dir = LaunchConfiguration("output_dir")
    enable_emotion = LaunchConfiguration("enable_emotion")

    return LaunchDescription(
        [
            DeclareLaunchArgument("config_path", default_value=default_config),
            DeclareLaunchArgument("output_dir", default_value="outputs/logs/ros2_playback_grounded"),
            DeclareLaunchArgument("enable_emotion", default_value="true"),
            DeclareLaunchArgument("playback_source_path", default_value=""),
            Node(package="social_robot", executable="playback_adapter_node", name="playback_adapter_node", output="screen", parameters=[{"config_path": config_path}]),
            Node(package="social_robot", executable="digital_twin_node", name="digital_twin_node", output="screen", parameters=[{"config_path": config_path}]),
            Node(
                package="social_robot",
                executable="emotion_inference_node",
                name="emotion_inference_node",
                output="screen",
                condition=IfCondition(enable_emotion),
                parameters=[{"config_path": config_path}],
            ),
            Node(
                package="social_robot",
                executable="event_logger_node",
                name="event_logger_node",
                output="screen",
                parameters=[{"config_path": config_path, "output_dir": output_dir}],
            ),
        ]
    )
