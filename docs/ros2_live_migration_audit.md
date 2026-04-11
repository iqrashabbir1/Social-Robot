# ROS 2 Live Migration Audit

## Existing Code Reused
- `src/ros2/interface_spec.py`: reusable Paper 1 topic specification seed.
- `src/ros2/playback_adapter.py`: reusable playback-grounded event mirroring logic.
- `src/ros2/bag_or_emulated_replay.py`: reusable rosbag-or-emulated topic source fallback.
- `src/digital_twin/twin_state.py`: reusable synchronized twin-state accumulator.
- `src/digital_twin/run_cs1_playback.py`: reusable playback-grounded CS1 validation path.
- `src/data/collect_pilot_session.py`: reusable live laptop-sensor collection path.
- `src/data/real_anchor_loader.py`: reusable real-anchor session loading.
- `perception/face_emotion.py`, `perception/speech_emotion.py`, `fusion/fusion_logic.py`: reusable lightweight live inference baseline.
- `src/hardware/live_validation.py`: reusable webcam/microphone availability logic for laptop-sensor validation.

## Suitable To Become ROS 2 Nodes
- Webcam capture path from `src/data/collect_pilot_session.py` -> `camera_node`
- Microphone capture path from `src/data/collect_pilot_session.py` / `src/hardware/live_validation.py` -> `audio_node`
- Placeholder context/system-health publishing from CS1/CS2 utilities -> `robot_state_node`
- Twin-state synchronization from `src/digital_twin/twin_state.py` -> `digital_twin_node`
- Baseline emotion pipeline from `perception/*` + `fusion/fusion_logic.py` -> `emotion_inference_node`
- Playback-grounded topic replay from `src/ros2/bag_or_emulated_replay.py` -> `playback_adapter_node`
- Structured CSV logging aligned with Paper 1 evidence labeling -> `event_logger_node`

## Should Remain Offline Utilities
- `src/models/classical/train_classical.py`
- `src/models/deep/train_deep_fusion.py`
- `src/models/transformer/train_transformer_fusion.py`
- `src/evaluation/benchmark_runner.py`
- `src/evaluation/export_results.py`
- `src/visualization/generate_all_figures.py`
- all Paper 1 benchmark configs under `configs/cs1`, `configs/cs2`, `configs/cs3`

## Playback-Grounded Code To Preserve
- `src/digital_twin/run_cs1_playback.py`
- `src/ros2/bag_or_emulated_replay.py`
- `src/ros2/playback_adapter.py`
- `configs/cs1/playback_grounded.yaml`
- Paper 1 playback-grounded docs and output labels

## Real-Anchor Code To Adapt
- `src/data/collect_pilot_session.py`
- `src/data/real_anchor_loader.py`
- `data/pilot/sessions/*`

## Existing ROS 2 Assets
- Legacy workspace skeleton: `ros2_ws/src/cognitive_caregiver/`
- Existing files: `package.xml`, `setup.py`, `setup.cfg`, `launch/caregiving_demo.launch.py`
- Assessment: useful as historical reference only; it does not match the Paper 1 topic graph and should not be the primary live package going forward.

## Missing Before This Migration
- Repo-root ROS 2 package files so the repository can live directly under `~/social_robot_ws/src/social_robot`
- Live ROS 2 node entrypoints matching the Paper 1 topic graph
- Launch files for live sensing, live emotion demo, playback-grounded runtime, and minimal demo
- ROS 2 oriented config files
- Rosbag record/replay workflow scripts and docs
- Live/offline architecture docs for WSL2 + Ubuntu 24.04 + ROS 2 Jazzy

## Recommended Paper 1 Live ROS 2 Graph
1. `camera_node` publishes `/camera/image_raw`
2. `audio_node` publishes `/audio/stream`
3. `robot_state_node` publishes `/robot_pose` and `/system_health`
4. `digital_twin_node` subscribes to sensing/state topics and publishes `/event_log`, `/head_cmd`, `/speech_cmd`
5. `emotion_inference_node` subscribes to live streams and publishes `/emotion_state`
6. `event_logger_node` subscribes to all critical topics and writes structured CSV evidence
7. `playback_adapter_node` can replace live sensor publishers with playback-grounded publishers while preserving downstream compatibility

## Migration Decision
- Keep the current Paper 1 offline benchmark stack unchanged.
- Keep playback-grounded CS1 as a first-class fallback.
- Add a new repo-root ROS 2 Python package named `social_robot`.
- Treat live laptop sensors as `ros2_live_laptop_sensors`, not as a deployed robot claim.
