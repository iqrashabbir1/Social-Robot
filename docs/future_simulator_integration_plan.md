# Future Simulator Integration Plan

## Goal
Swap laptop-sensor publishers with simulator publishers while keeping downstream nodes unchanged.

## Candidate Simulators
- Gazebo
- Webots

## Substitution Strategy
- replace `camera_node` with simulator image publisher on `/camera/image_raw`
- replace `audio_node` with simulator or synthetic audio publisher on `/audio/stream`
- replace `robot_state_node` with simulator pose/state publisher on `/robot_pose` and `/system_health`
- keep `digital_twin_node`, `emotion_inference_node`, and `event_logger_node` unchanged

## Paper 1 Benefit
The current topic graph is already structured so that simulator integration becomes a publisher swap, not a full downstream rewrite.
