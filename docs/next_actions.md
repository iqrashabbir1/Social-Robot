# Next Actions

## What Already Existed
- `perception/` with face and speech emotion modules
- `fusion/` with rule-based multimodal fusion logic
- `response/` with empathy response rules and TTS support
- `tests/` with webcam, speech, and demo scripts
- `data/` with CREMA-D speech assets, user SER data, trained shallow models, and temporary recordings
- `ros2_ws/` and `paper/` directories as early placeholders

## What Was Reused
- DeepFace-based webcam emotion pipeline as the preserved baseline MER path
- MFCC feature extraction and SVM speech emotion classification assets
- fusion and response rules as baseline interaction logic
- existing speech datasets and trained models as reusable baseline resources

## What Was Newly Added
- publication-oriented workspace folders under `docs/`, `src/`, `configs/`, `outputs/`, and `literature/`
- research framing and problem formulation documents
- system architecture document and module map
- eight case-study design files with CSV-backed metrics and summaries
- literature comparison matrix and summary package
- Python scaffolding for benchmarks, ablations, CSV export, and figure generation
- research smoke tests for core metrics utilities

## What Is Still Missing For Full Experiments
- a standard Python runtime in the active shell for executing the new scripts
- synchronized multimodal datasets that include physiology, motion, and medication context
- ROS2 digital-twin implementation details and scenario logs
- dashboard frontend and telepresence integration
- KG population and explanation quality annotation data
- pilot-study protocol, ethics path, and deployment hardware plan

## Recommended Immediate Next Steps
1. Install or expose a standard Python runtime in the shell so the new benchmark and plotting scripts can be executed.
2. Validate the preserved baseline end-to-end and export standardized benchmark logs.
3. Populate the digital-twin and physiology simulation feeds for CS1, CS2, CS4, and CS8.
4. Decide the first publication target:
   - MER benchmark paper
   - integrated architecture paper
   - explainability and oversight paper
5. Add a lightweight caregiver dashboard prototype to ground CS6 and CS7 with executable evidence.
