# CS2: Multimodal Sensing and Data Synchronization

## Purpose
Evaluate whether speech, visual, behavioral, and future physiological signals can be aligned robustly enough for downstream risk and interaction tasks.

## Dataset, Simulator, and Input Assumptions
- Existing webcam and speech assets act as implemented modalities.
- Additional movement and physiology streams are simulated for synchronization stress tests.
- Quality flags mark missing or delayed packets.

## Method
- Define a shared windowing and timestamp policy.
- Compare naive nearest-neighbor alignment with explicit synchronization buffers and missing-modality flags.
- Assess degradation when one or more modalities arrive late or drop out.

## Baselines
- A1: Classical wearable monitoring plus shallow ML risk prediction
- A5: Multimodal emotion recognition only
- A8: Proposed integrated system

## Metrics
- synchronization error
- packet completeness
- fused window coverage
- missing-modality robustness
- ingestion latency

## Expected Findings
- Explicit synchronization logic should outperform ad hoc fusion for unstable multimodal streams.
- The proposed stack should support richer downstream tasks because it retains quality metadata, not just raw aligned values.

## Failure Modes
- timestamp drift
- modality dominance from higher-rate streams
- silent failure when a stream is missing
- unreliable late-arriving windows

## Journal-Quality Figure Plan
- modality timeline alignment diagram
- synchronization error bars
- robustness under dropout chart
