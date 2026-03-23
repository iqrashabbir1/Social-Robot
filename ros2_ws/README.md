# ROS2 Workspace

This workspace now includes a lightweight ROS2 package skeleton for the cognitive caregiving robot.

## Intended Nodes
- `digital_twin_node`
- `risk_router_node`

These nodes mirror the repository's research architecture and topic schema:
- `/caregiver/sensors/visual`
- `/caregiver/sensors/audio`
- `/caregiver/sensors/physiology`
- `/caregiver/digital_twin/state`
- `/caregiver/alerts/risk`
- `/caregiver/alerts/adherence`
- `/caregiver/dashboard/override`
- `/caregiver/telepresence/session`

The Python-side research pipeline in `src/` remains the main executable benchmark path. The ROS2 package is a workspace-ready bridge for future simulation and deployment work.
