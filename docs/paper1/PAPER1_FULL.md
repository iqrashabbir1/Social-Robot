# Paper 1 Full Manuscript

## Title
A Simulation-First ROS2-Compatible Digital-Twin and Multimodal Benchmarking Framework for Cognitive Caregiving Robots

## Abstract
This paper presents a simulation-first, ROS2-compatible, technically validated, and preliminary framework for cognitive caregiving robots, with emphasis on synchronization, robustness, and multimodal emotion benchmarking. The implementation is organized around three measurable case studies. CS1 evaluates a digital-twin backbone through ROS2-compatible topic interfaces, timestamped event logging, playback grounding, and disturbance analysis. CS2 builds a multimodal sensing pipeline that aligns video, audio, robot-context, and optional physiology placeholders into fixed windows while explicitly tracking modality availability, and now includes a small local pilot real-anchor session. CS3 preserves the repository’s existing visual baseline and adds preliminary benchmark paths for classical, deep-fusion, and transformer-style multimodal models under clearly labeled synthetic or pilot-demonstration conditions.

The contribution is not a claim of clinical efficacy, live robot deployment, or validated caregiving outcomes. Instead, the paper establishes a reproducible technical backbone that supports later simulator studies, richer multimodal data collection, and future ROS2-integrated robot experiments. All outputs are exported as CSV files, figures are generated from those CSVs, and results are labeled as synthetic, pilot real-anchor, playback-grounded, or mixed.

## 1. Introduction
Cognitive caregiving robots require more than perception models. They also require a measurable runtime backbone that can synchronize sensing, mirror system state in a digital twin, and support controlled replay and disturbance analysis. This repository originally contained baseline social-robot perception code, but not a clean first-paper experiment path. Paper 1 addresses that gap by reorganizing the codebase around three foundational case studies: ROS2-compatible digital-twin validation, multimodal sensing and synchronization, and preliminary emotion-recognition benchmarking.

The resulting contribution is a technical framework rather than a deployment claim. The repository now separates synthetic benchmarking, pilot real-anchor evidence, and playback-grounded system validation so the maturity of each component is explicit. This keeps the present evidence defensible while creating a rigorous basis for later simulator integration and real robot studies.

## 2. Related Work
Prior work on social and caregiving robots often emphasizes interaction goals, behavior generation, or application framing, but less often provides a measurable synchronization and digital-twin backbone. Digital twins in robotics are increasingly used for monitoring and replay, yet these studies do not always focus on multimodal HRI sensing. Multimodal emotion recognition has also advanced through deep and transformer-style fusion, but many benchmarks remain detached from robot runtime constraints. Paper 1 integrates these strands by treating ROS2-compatible interfaces, synchronization, playback grounding, and emotion benchmarking as one coherent experimental substrate.

## 3. System Architecture
The Paper 1 architecture contains three layers. First, a ROS2-compatible topic specification defines the main sensing and control interfaces: `/camera/image_raw`, `/audio/stream`, `/robot_pose`, `/head_cmd`, `/speech_cmd`, `/event_log`, and `/system_health`. Second, a digital twin mirrors timestamped topic activity and supports replay, playback grounding, and disturbance analysis. Third, a multimodal synchronization and benchmark stack constructs fixed windows and evaluates emotion-recognition model families.

The preserved repository baseline is retained as Baseline-0. New Paper 1 code lives under `src/common/`, `src/ros2/`, `src/digital_twin/`, `src/data/`, `src/features/`, and benchmark-specific model and visualization modules.

## 4. Methodology
CS1 evaluates simulator-only validation and playback-grounded validation using ROS2-compatible topics. On this machine, the playback-grounded path uses a ROS2-compatible emulation layer because the native `ros2` CLI is not on `PATH`. CS2 constructs aligned video, audio, robot-context, and physiology-placeholder streams, then builds fixed windows with explicit modality-availability tracking. The updated implementation also supports a small locally collected pilot real-anchor session. CS3 evaluates one preserved baseline and multiple preliminary benchmark families.

The current repository evidence supports a 4-class label space of `happy`, `sad`, `neutral`, and `fear`. Because `angry` is not present in the preserved baseline log, it is treated as future work rather than backfilled synthetically. Baseline-0 is evaluated from the preserved visual log and can also be run in pilot real-anchor inference mode. Trainable B1 through B3 comparisons remain preliminary because they rely mainly on synthetic aligned multimodal windows designed to test the benchmark pipeline itself rather than to claim field performance.

## 5. Results
CS1 now contains both simulation-first and playback-grounded outputs. The playback-grounded path exports latency metrics, synchronization traces, and event timing using ROS2-compatible message replay. CS2 now contains both synthetic synchronization studies and a pilot real-anchor session collected on the local machine using webcam, microphone, and a timestamped context log. CS3 includes the preserved baseline, an enabled XGBoost classical configuration, and preliminary classical/deep/transformer comparisons on synthetic aligned windows, plus a pilot real-anchor baseline inference demonstration.

These results should be interpreted as framework validation and pilot demonstrations, not as evidence of real caregiving effectiveness or live deployment readiness.

## 6. Discussion
The main success of Paper 1 is structural. The repository now cleanly separates preserved baseline evidence, pilot real-anchor evidence, playback-grounded validation, and synthetic benchmark studies. This is important for publication quality because it avoids mixing real and simulated claims. The main limitation is that the pilot anchor is intentionally small, the ROS2 path is playback-grounded rather than live, and the deep and transformer comparisons still rely primarily on synthetic multimodal data.

## 7. Conclusion
Paper 1 turns the repository into a reproducible, simulation-first foundation for digital-twin validation, synchronized multimodal sensing, and preliminary emotion-aware benchmarking. The framework is ready for a first paper focused on synchronization, robustness, playback grounding, and benchmark readiness, while also documenting the exact next steps required for live ROS2 integration, larger real multimodal datasets, and future caregiving intelligence studies.

## Notes on Evidence
- `B0` is a preserved baseline and now also supports a pilot real-anchor inference demonstration.
- `B1`, `B2`, and `B3` remain preliminary multimodal benchmarks, with most trainable comparisons still driven by synthetic aligned windows.
- CS1 is simulation-first and playback-grounded, not a live ROS2 robot deployment study.
- CS2 includes both synthetic synchronization studies and a small pilot real-anchor pathway.
