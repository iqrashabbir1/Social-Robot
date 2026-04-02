# Paper 1 Full Manuscript

## Title
A ROS2 Digital-Twin Framework for Multimodal Cognitive Caregiving Robots: Synchronization, Robustness, and Emotion-Aware Benchmarking

## Abstract
This paper presents a ROS2 digital-twin framework for multimodal cognitive caregiving robots, with emphasis on synchronization, robustness, and emotion-aware benchmarking. The current repository is reorganized around three measurable case studies. CS1 evaluates a ROS2-aligned digital-twin backbone through software-equivalent topic interfaces, timestamped event logging, replay, and fault injection, reporting latency, synchronization error, message drop rate, task success, recovery rate, and resource usage from reproducible simulation-backed runs. CS2 builds a multimodal sensing pipeline that aligns video, audio, robot-context, and optional physiology placeholders into fixed windows while explicitly tracking modality availability and missing-modality robustness. CS3 preserves the repository’s existing real visual emotion baseline as Baseline-0 and adds benchmark scaffolding for classical, deep-fusion, and transformer-style multimodal models under clearly labeled synthetic placeholder conditions.

The contribution is not a claim of clinical deployment or field performance. Instead, the paper establishes a reproducible experimental backbone that makes later caregiving intelligence modules measurable and publishable. The resulting workspace exports all results as CSV files, generates publication-style figures directly from those CSVs, and separates real baseline evidence from synthetic multimodal placeholders. This produces a practical, research-grade foundation for future ROS2 simulation studies, synchronized dataset collection, and real human-robot interaction benchmarking in caregiving environments.

## 1. Introduction
Cognitive caregiving robots require more than perception models. They also require a measurable runtime backbone that can synchronize sensing, mirror system state in a digital twin, and support controlled replay and disturbance analysis. This repository originally contained baseline social-robot perception code, but not a clean first-paper experiment path. Paper 1 addresses that gap by reorganizing the codebase around three foundational case studies: ROS2-plus-digital-twin validation, multimodal sensing and synchronization, and emotion-recognition benchmarking.

The resulting contribution is a publication-oriented framework rather than a deployment claim. The repository preserves an implemented real visual baseline, introduces software-equivalent digital-twin and synchronization studies, and adds synthetic placeholder multimodal benchmarking for later model families. This separation makes the current evidence explicit while still creating a rigorous foundation for future caregiving intelligence modules.

## 2. Related Work
Prior work on social and caregiving robots often emphasizes interaction goals, prompting behavior, or application framing, but less often provides a measurable synchronization and digital-twin backbone. Digital twins in robotics are increasingly used for monitoring and replay, yet these studies do not always focus on multimodal HRI sensing. Multimodal emotion recognition has also advanced through deep and transformer-style fusion, but many benchmarks remain detached from robot runtime constraints. Paper 1 integrates these strands by treating ROS2 interfaces, synchronization, and emotion benchmarking as one coherent experimental substrate.

## 3. System Architecture
The Paper 1 architecture contains three layers. First, a ROS2-compatible topic specification defines the main sensing and control interfaces: `/camera/image_raw`, `/audio/stream`, `/robot_pose`, `/head_cmd`, `/speech_cmd`, `/event_log`, and `/system_health`. Second, a digital twin mirrors timestamped topic activity and supports replay and fault injection. Third, a multimodal synchronization and benchmark stack constructs fixed windows and evaluates emotion-recognition model families.

The preserved repository baseline is retained as Baseline-0. New Paper 1 code lives under `src/common/`, `src/ros2/`, `src/digital_twin/`, `src/data/`, `src/features/`, and benchmark-specific model and visualization modules.

## 4. Methodology
CS1 evaluates four experiment modes: simulator only, simulator plus control loop, simulator plus playback, and simulator plus injected faults. The outputs include latency metrics, synchronization error traces, and fault robustness summaries. CS2 constructs aligned video, audio, robot-context, and physiology-placeholder streams, then builds fixed windows with explicit modality-availability tracking. CS3 evaluates one preserved baseline and three benchmark families.

The current repository evidence supports a 4-class label space of `happy`, `sad`, `neutral`, and `fear`. Because `angry` is not present in the preserved baseline log, it is treated as future work rather than backfilled synthetically. Baseline-0 is evaluated from the real visual log. B1 through B3 are executed on a synthetic placeholder multimodal dataset derived to test the benchmark pipeline itself rather than to claim field performance.

## 5. Results
In CS1, mean latency increases from `19.1193 ms` in M1 to `40.4978 ms` in M4, with M4 also showing a `0.0531` message drop rate and `0.9459` task success rate. This demonstrates that the scaffold captures measurable degradation under disturbance. In CS2, the aligned nominal condition reports mean alignment error of `52.6608 ms`, while missing-modality stress raises this to `63.1930 ms` and reduces the full-modality window rate from `0.5072` to `0.4159`.

In CS3, the strongest real evidence remains the preserved visual baseline with accuracy `0.8350` and macro F1 `0.8155`. The multimodal benchmark families are currently synthetic placeholders: classical SVM, deep late fusion, and transformer-style fusion. Their outputs demonstrate executable benchmarking infrastructure and ablation support, but they must not be interpreted as real-world caregiving performance claims.

## 6. Discussion
The main success of Paper 1 is structural. The repository now cleanly separates preserved baseline evidence from synthetic multimodal placeholder studies and from future deployment tasks. This is important for publication quality because it avoids mixing real and simulated claims. The main limitation is that aligned real multimodal caregiving data are not yet available in the repository, and ROS2 integration remains software-equivalent rather than bound to a full simulator or robot runtime.

## 7. Conclusion
Paper 1 turns the repository into a reproducible foundation for digital-twin validation, synchronized multimodal sensing, and emotion-aware benchmarking. The framework is ready for a first paper focused on synchronization, robustness, and benchmark readiness, while also documenting the exact next steps required for real simulator integration, aligned data collection, and future caregiving intelligence modules.

## Notes on Evidence
- `B0` is the preserved implemented real baseline.
- `B1`, `B2`, and `B3` are current synthetic placeholder multimodal benchmarks.
- CS1 is a simulation-backed system study.
- CS2 is a synchronized placeholder pipeline ready for replacement with real multimodal session data.
