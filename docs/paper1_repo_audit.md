# Paper 1 Repository Audit

## Scope of This Audit
This audit reviews the current repository only for Paper 1:

- CS1: ROS2 plus digital twin validation
- CS2: multimodal sensing and synchronization
- CS3: emotion recognition benchmarking

Later-stage modules such as medication adherence, knowledge-graph explanation, HITL dashboarding, privacy, and telepresence are treated as out of scope for Paper 1 except where the codebase should remain extensible.

## Existing Folders and Key Files

### Top-Level Folders
- `configs/`
- `data/`
- `docs/`
- `fusion/`
- `literature/`
- `notebooks/`
- `outputs/`
- `paper/`
- `perception/`
- `response/`
- `ros2_ws/`
- `src/`
- `tests/`

### Existing Emotion-Recognition Code
- `perception/face_emotion.py`
  - DeepFace-based single-frame facial emotion inference.
  - Reusable as the preserved visual baseline for Paper 1.
- `perception/speech_features.py`
  - MFCC feature extraction for audio clips.
  - Reusable for audio feature extraction and classical audio baselines.
- `perception/speech_emotion.py`
  - SVM-based speech emotion inference using saved model assets.
  - Reusable for audio-only benchmarking support, but the current label space is 3-class and not directly aligned with the 4-class visual baseline.
- `fusion/fusion_logic.py`
  - Rule-based emotion fusion logic.
  - Useful for baseline preservation, but not sufficient as the main benchmark engine for Paper 1.
- `src/pipelines/baseline_mer_pipeline.py`
  - Already documents the existing baseline stack as a reusable MER reference.

### Existing ROS2 and Simulation Assets
- `ros2_ws/src/cognitive_caregiver/cognitive_caregiver/digital_twin_node.py`
- `ros2_ws/src/cognitive_caregiver/cognitive_caregiver/physiology_replay_node.py`
- `ros2_ws/src/cognitive_caregiver/cognitive_caregiver/risk_router_node.py`
- `ros2_ws/src/cognitive_caregiver/launch/caregiving_demo.launch.py`
- `src/pipelines/digital_twin_orchestrator.py`

These files show that the repo already contains:
- a ROS2 workspace skeleton
- a basic digital-twin topic concept
- replay-style node stubs

However, they are not yet organized around Paper 1’s required CS1 interfaces, synchronization metrics, replay study modes, or robustness benchmarks.

### Existing Evaluation and Plotting Assets
- `src/evaluation/metrics.py`
  - Useful classification and latency summary helpers.
- `src/evaluation/run_benchmarks.py`
  - Broad benchmark script, but it mixes Paper 1 with later-paper topics.
- `src/visualization/plot_style.py`
  - Reusable publication-style plotting defaults.
- `src/visualization/generate_all_figures.py`
  - Existing figure generator, but it targets the broader multi-paper repository state rather than Paper 1 only.

### Existing Data and Reusable Assets
- `tests/emotion_log_labeled.csv`
  - Reusable 4-class visual baseline evaluation log.
- `data/speech/speech_dataset_crema.npz`
  - Reusable speech dataset asset.
- `data/speech/speech_svm_crema_balanced.joblib`
  - Reusable classical speech model artifact.
- `tests/build_speech_dataset_crema.py`
  - Reusable data-preparation reference.
- `tests/evaluate_vision_module.py`
  - Reusable evaluation reference for the visual baseline.

## What Is Reusable for Paper 1

### Keep
- `perception/face_emotion.py`
- `perception/speech_features.py`
- `perception/speech_emotion.py`
- `src/pipelines/baseline_mer_pipeline.py`
- `src/evaluation/metrics.py`
- `src/visualization/plot_style.py`
- ROS2 workspace skeleton under `ros2_ws/`
- labeled vision baseline log under `tests/emotion_log_labeled.csv`

### Refactor Around
- `src/evaluation/run_benchmarks.py`
  - Too broad for Paper 1, but contains useful evaluation/export patterns.
- `src/visualization/generate_all_figures.py`
  - Needs a Paper 1-specific figure pipeline.
- `src/pipelines/digital_twin_orchestrator.py`
  - Conceptually related, but too high-level and not aligned with CS1 metrics and interfaces.

### Deprecate for Paper 1 Execution Path
These files are not removed, but they should not be treated as the main Paper 1 path:
- later case-study docs in `docs/case_studies/` beyond CS1–CS3
- outputs under `outputs/csv/` and `outputs/figures/` that belong to later modules
- broader pipeline components for medication, explainability, dashboarding, and pilot readiness

## Missing Pieces for Paper 1

### CS1 Gaps
- formal Paper 1 ROS2 interface specification
- event logger aligned to required Paper 1 topics
- replay runner for controlled experiment modes
- digital twin state mirror tied to synchronization metrics
- fault injection utilities
- CSV exports for latency, sync error, fault robustness, and resource usage

### CS2 Gaps
- explicit multimodal timestamp alignment pipeline
- modality availability tracker
- synchronized fixed-window builder
- missing-modality handling utilities
- synchronization quality metrics and exports

### CS3 Gaps
- Paper 1-specific benchmark runner limited to B0/B1/B2/B3
- preserved B0 baseline integration in the new result format
- 4-class benchmark framing that separates:
  - implemented real baseline
  - synthetic placeholder multimodal benchmark
  - TODO for future real aligned multimodal data collection
- ablation runner for modality subsets
- inference latency benchmarking
- Paper 1-only plots and tables

### Documentation Gaps
- Paper 1-only audit, methodology, figure plan, table plan, and run guide
- first full Paper 1 markdown draft
- final Paper 1 status report

## Decisions

### Keep
- existing baseline facial emotion inference
- existing speech feature extraction and speech SVM assets
- existing plotting style
- existing ROS2 workspace skeleton

### Add
- a Paper 1-specific experiment structure under:
  - `docs/paper1/`
  - `src/common/`
  - `src/ros2/`
  - `src/digital_twin/`
  - `src/data/` additions for synchronization
  - `src/features/`
  - `src/models/classical/`
  - `src/models/deep/`
  - `src/models/transformer/`
- Paper 1 configs under `configs/cs1/`, `configs/cs2/`, and `configs/cs3/`
- Paper 1 CSV and figure outputs under `outputs/csv/cs1/`, `outputs/csv/cs2/`, `outputs/csv/cs3/`, and matching figure folders

### Refactor
- plotting entrypoints so Paper 1 figures can be generated independently from later-paper content
- evaluation logic so CS1–CS3 are reproducible and isolated

### Do Not Remove
- existing baseline perception code
- ROS2 workspace assets
- broad repository documents and outputs that may support later papers

## Paper 1 Implementation Strategy
1. Preserve the current DeepFace-style visual baseline as Baseline-0.
2. Build a clean Paper 1 experiment path around CS1, CS2, and CS3 only.
3. Use real existing assets where available:
   - visual baseline logs
   - speech MFCC extraction
   - speech model/data references
   - ROS2 skeleton
4. Use clearly labeled synthetic placeholder data where aligned multimodal data are currently missing.
5. Export all Paper 1 results to CSV first, then generate publication-style figures only from those CSVs.
6. Draft the paper directly from the executable outputs and explicit evidence labels.
