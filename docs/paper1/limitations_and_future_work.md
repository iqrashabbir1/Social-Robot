# Limitations and Future Work

## Current limitations
- The pilot real-anchor dataset is intentionally small and locally collected.
- The ROS2 runtime on this machine uses playback-compatible emulation because `ros2` is not on `PATH`.
- Deep and transformer comparisons remain mostly synthetic.
- Real robot actuation, telepresence, medication support, and clinical workflows are outside Paper 1 scope.

## Near-term next steps
- connect CS1 to a full ROS2 simulator or rosbag2 runtime
- expand pilot sessions into a modest multimodal benchmark set
- add real weak labels or controlled prompt labels for the real-anchor data
- evaluate domain shift from synthetic training to pilot real-anchor inference
- add mini-batch and checkpointed training to a dedicated real-data branch when more data are available

## Long-term future work
- live simulator integration
- real robot middleware validation
- larger HRI studies
- later caregiving modules such as adherence, oversight, and explainability
