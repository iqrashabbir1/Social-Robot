# Results Caption Drafts

## Hybrid architecture
Figure X shows the verified Paper 1 hybrid runtime, in which webcam sensing is captured natively on Windows, forwarded as a TCP image stream, and republished inside the WSL ROS 2 graph for digital-twin, inference, logging, and rosbag workflows.

## Runtime verification
Figure Y summarizes the portions of the hybrid runtime that have been directly verified in the current project state, while preserving the claim boundary that this remains a laptop-sensor platform demonstration rather than a deployed caregiving robot. This figure is better treated as supplementary if the main paper needs a tighter results section.

## Hybrid camera sample
Do not use the current local output in the paper. Replace it only after exporting true hybrid runtime frames from `/camera/image_raw` or a rosbag-derived frame set.

## Hybrid camera frame rate
Do not use the current local output in the paper. Replace it only after exporting a measured hybrid `ros2_event_log.csv` or rosbag-derived timestamp series.

## System health
Do not use the current local output in the paper. Replace it only after exporting a measured hybrid `ros2_system_health.csv`.

## Runtime mode comparison
Figure C compares playback-grounded, legacy live laptop sensing, and hybrid Windows-stream operation across verification status, topic availability, rosbag support, and robustness to host-side camera constraints. Use it as supplementary framing or convert it into a results table rather than a primary results figure.

## Dataset sample panel
Figure D presents representative samples from the controlled Paper 1 image-set evaluation path. Unlike ad hoc live captures, these inputs are replayable and can be reused for consistent benchmarking and qualitative inspection.

## Dataset prediction panel
Figure E shows predicted emotion labels on fixed dataset inputs. Use this as a main qualitative figure only when the underlying dataset is itself publication-appropriate and labeled; the currently tracked local room-scene pilot set is better treated as internal or supplementary only.

## Dataset replay sequence
Figure F illustrates dataset frames replayed through the ROS2 pipeline on `/camera/image_raw`, demonstrating that the same downstream graph can consume controlled dataset input and live hybrid input.

## Dataset confusion matrix
Figure G reports the confusion matrix when explicit labels are available through folder structure or a label CSV. If the local dataset is unlabeled, this figure should not be used in the manuscript.

## Dataset metrics
Figure H summarizes offline dataset-evaluation metrics. When labels are unavailable, only coverage and confidence statistics are plotted and should be reported as preliminary.
