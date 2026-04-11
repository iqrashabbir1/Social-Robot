from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    default_config = PathJoinSubstitution([FindPackageShare("social_robot"), "config", "dataset_replay.yaml"])
    config_path = LaunchConfiguration("config_path")
    output_dir = LaunchConfiguration("output_dir")
    enable_emotion = LaunchConfiguration("enable_emotion")
    runtime_type = LaunchConfiguration("runtime_type")

    return LaunchDescription(
        [
            DeclareLaunchArgument("config_path", default_value=default_config),
            DeclareLaunchArgument("output_dir", default_value="outputs/logs/ros2_dataset_replay"),
            DeclareLaunchArgument("enable_emotion", default_value="true"),
            DeclareLaunchArgument("runtime_type", default_value="ros2_dataset_replay"),
            Node(
                package="social_robot",
                executable="dataset_replay_node",
                name="dataset_replay_node",
                output="screen",
                parameters=[{"config_path": config_path, "runtime_type": runtime_type}],
            ),
            Node(
                package="social_robot",
                executable="robot_state_node",
                name="robot_state_node",
                output="screen",
                parameters=[{"config_path": config_path, "runtime_type": runtime_type}],
            ),
            Node(
                package="social_robot",
                executable="digital_twin_node",
                name="digital_twin_node",
                output="screen",
                parameters=[{"config_path": config_path, "runtime_type": runtime_type}],
            ),
            Node(
                package="social_robot",
                executable="emotion_inference_node",
                name="emotion_inference_node",
                output="screen",
                condition=IfCondition(enable_emotion),
                parameters=[{"config_path": config_path, "runtime_type": runtime_type}],
            ),
            Node(
                package="social_robot",
                executable="event_logger_node",
                name="event_logger_node",
                output="screen",
                parameters=[{"config_path": config_path, "output_dir": output_dir, "runtime_type": runtime_type}],
            ),
        ]
    )
