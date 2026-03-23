# CS3: Classical ML Versus Deep Multimodal Versus Transformer MER Benchmark

## Purpose
Benchmark the preserved baseline MER path against stronger multimodal variants while avoiding fabricated results.

## Dataset, Simulator, and Input Assumptions
- Existing assets: DeepFace-based webcam pipeline, CREMA-D speech data, user SER samples.
- Future extensions: synchronized multimodal emotion clips with elderly-facing interaction context.
- Standardized benchmark splits are still to be finalized.

## Method
- Use the current DeepFace plus speech SVM pipeline as the implemented baseline.
- Define late-fusion and transformer baselines as reproducible experimental targets.
- Report current baseline readiness separately from planned deep-model experiments.

## Baselines
- A5: Multimodal emotion recognition system only
- A8: Proposed integrated system

## Metrics
- accuracy
- macro F1
- confusion matrix
- calibration error
- latency
- missing-modality robustness

## Expected Findings
- The current baseline is expected to remain the strongest immediately reproducible reference.
- Later multimodal deep models should improve robustness and uncertainty handling when synchronized data are available.

## Failure Modes
- dataset shift between public speech data and target elderly domain
- label imbalance
- over-reliance on facial priors
- poor confidence calibration

## Journal-Quality Figure Plan
- confusion matrices
- ablation bars
- calibration plot
- latency versus utility figure
