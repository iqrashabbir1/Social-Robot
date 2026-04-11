# How To Build The ROS 2 Workspace

## Build
```bash
cd ~/social_robot_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
```

## Source The Workspace
```bash
source ~/social_robot_ws/install/setup.bash
```

## Verify Package Discovery
```bash
ros2 pkg list | grep social_robot
ros2 pkg prefix social_robot
```

## Recommended Workflow
1. Source ROS 2 Jazzy
2. Build with `colcon`
3. Source the workspace
4. Launch one of the Paper 1 graphs

## Important
- Do not run ROS 2 launch files from an unsourced shell.
- Do not leave Conda active in the same shell used for ROS 2 build/launch.
