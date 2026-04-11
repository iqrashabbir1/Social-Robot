from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    default_config = PathJoinSubstitution([FindPackageShare("social_robot"), "config", "paper1_minimal.yaml"])
    config_path = LaunchConfiguration("config_path")
    output_dir = LaunchConfiguration("output_dir")
    enable_audio = LaunchConfiguration("enable_audio")
    enable_emotion = LaunchConfiguration("enable_emotion")
    camera_input_mode = LaunchConfiguration("camera_input_mode")
    runtime_type = LaunchConfiguration("runtime_type")
    windows_camera_host = LaunchConfiguration("windows_camera_host")
    windows_camera_port = LaunchConfiguration("windows_camera_port")

    return LaunchDescription(
        [
            DeclareLaunchArgument("config_path", default_value=default_config),
            DeclareLaunchArgument("output_dir", default_value="outputs/logs/paper1_minimal"),
            DeclareLaunchArgument("enable_audio", default_value="true"),
            DeclareLaunchArgument("enable_emotion", default_value="true"),
            DeclareLaunchArgument("camera_input_mode", default_value="windows_stream_bridge"),
            DeclareLaunchArgument("runtime_type", default_value="ros2_live_windows_stream_wsl_core"),
            DeclareLaunchArgument("windows_camera_host", default_value="127.0.0.1"),
            DeclareLaunchArgument("windows_camera_port", default_value="5001"),
            Node(
                package="social_robot",
                executable="camera_node",
                name="camera_node",
                output="screen",
                parameters=[
                    {
                        "config_path": config_path,
                        "camera_input_mode": camera_input_mode,
                        "windows_camera_host": windows_camera_host,
                        "windows_camera_port": windows_camera_port,
                    }
                ],
            ),
            Node(
                package="social_robot",
                executable="audio_node",
                name="audio_node",
                output="screen",
                condition=IfCondition(enable_audio),
                parameters=[{"config_path": config_path}],
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
