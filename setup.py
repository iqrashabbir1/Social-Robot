from setuptools import find_packages, setup


package_name = "social_robot"


setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(
        include=[
            "social_robot",
            "social_robot.*",
            "src",
            "src.*",
            "perception",
            "perception.*",
            "fusion",
            "fusion.*",
        ]
    ),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", [
            "launch/live_sensing.launch.py",
            "launch/live_emotion_demo.launch.py",
            "launch/playback_grounded.launch.py",
            "launch/paper1_minimal.launch.py",
            "launch/dataset_replay.launch.py",
        ]),
        (f"share/{package_name}/config", [
            "config/live_sensing.yaml",
            "config/live_emotion_demo.yaml",
            "config/playback_grounded.yaml",
            "config/paper1_minimal.yaml",
            "config/dataset_replay.yaml",
            "config/dataset_replay_rafdb.yaml",
            "config/dataset_replay_cremad.yaml",
            "config/dataset_replay_ravdess.yaml",
            "config/dataset_replay_emorynlp.yaml",
        ]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="iqrashabbir1",
    maintainer_email="iqraaaaashabbir@gmail.com",
    description="ROS 2 live-runtime package for Paper 1 multimodal sensing, digital twin, and emotion demo flows.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "camera_node = social_robot.camera_node:main",
            "audio_node = social_robot.audio_node:main",
            "robot_state_node = social_robot.robot_state_node:main",
            "digital_twin_node = social_robot.digital_twin_node:main",
            "emotion_inference_node = social_robot.emotion_inference_node:main",
            "event_logger_node = social_robot.event_logger_node:main",
            "windows_camera_bridge_node = social_robot.windows_camera_bridge_node:main",
            "playback_adapter_node = social_robot.playback_adapter_node:main",
            "dataset_replay_node = social_robot.dataset_replay_node:main",
        ],
    },
)
