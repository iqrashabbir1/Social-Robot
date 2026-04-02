# Methodology

## Overall Design
Paper 1 is organized around three case studies:
- CS1 validates the ROS2-plus-digital-twin backbone through software-equivalent interfaces, event logging, replay, and fault injection.
- CS2 constructs the multimodal sensing and synchronization pipeline with fixed temporal windows and explicit modality-availability tracking.
- CS3 benchmarks emotion-recognition model families while preserving the existing repository baseline.

## Evidence Separation
- `implemented_real_baseline`: existing visual baseline evaluated from `tests/emotion_log_labeled.csv`
- `synthetic_placeholder_benchmark`: aligned multimodal benchmark runs used when real synchronized multimodal caregiving data are not yet available
- `simulation_based_evaluation`: digital-twin runtime and robustness studies executed in software-equivalent form

## Label Space Note
The currently available 4-class repository evidence is `happy`, `sad`, `neutral`, and `fear`. The originally intended `angry` class is not supported by the preserved baseline log and is therefore not claimed in Paper 1. Extending the label space to `angry` is a future data-collection task rather than a synthetic relabeling step.

## CS1
CS1 simulates four experiment modes: simulator-only, simulator plus control loop, simulator plus playback, and simulator plus injected faults. The system tracks end-to-end latency, synchronization error, message drop rate, task success rate, recovery rate, and resource usage across the required interface set.

## CS2
CS2 aligns video, audio, robot context, and physiology-placeholder streams with timestamp tolerance and fixed windows. The pipeline records window indices, modality availability ratios, and synchronization quality under nominal and missing-modality stress conditions.

## CS3
CS3 uses Baseline-0 from the existing visual pipeline and adds three benchmark families for multimodal comparison:
- B1 classical SVM-based fusion
- B2 deep late-fusion MLP
- B3 transformer-style lightweight cross-modal fusion

All results are exported to CSV before figure generation. The synthetic multimodal dataset is explicitly marked as placeholder benchmark infrastructure, not a claimed field dataset.
