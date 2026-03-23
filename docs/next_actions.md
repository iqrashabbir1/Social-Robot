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

## What Is Now Executable
- a project-local Python runtime via `.venv`
- benchmark execution through `src/evaluation/run_benchmarks.py`
- figure generation including PDF export through `src/visualization/generate_all_figures.py`
- synthetic physiology, adherence, and alert generation
- populated care knowledge graph export
- lightweight dashboard generation at `outputs/dashboard/index.html`
- pilot-style validation protocol and readiness table
- ROS2 package skeleton in `ros2_ws/src/cognitive_caregiver`

## External Steps Still Required
1. Validate the preserved baseline with live webcam and microphone hardware on the target machine.
2. Connect real physiological or wearable streams to replace the current simulation-backed adapters.
3. Integrate the ROS2 package into an actual simulation or robot stack.
4. Decide the first publication target:
   - MER benchmark paper
   - integrated architecture paper
   - explainability and oversight paper
5. Complete ethics, site approval, and clinician coordination for a real pilot study.
