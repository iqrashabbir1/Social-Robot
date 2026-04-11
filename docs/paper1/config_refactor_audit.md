# Paper 1 Config Refactor Audit

## Current repository state

### Existing folders and key files reused
- `perception/face_emotion.py`: preserved webcam/DeepFace-style visual emotion baseline.
- `perception/speech_emotion.py`: preserved audio emotion baseline path.
- `fusion/fusion_logic.py`: preserved legacy fusion logic from the original repo.
- `src/digital_twin/run_cs1.py`: already implemented a synthetic CS1 simulator and topic logging path.
- `src/data/run_cs2.py`: already implemented a synthetic synchronization pipeline and window builder.
- `src/models/classical/train_classical.py`: already implemented classical feature-based benchmarking primitives.
- `src/models/deep/train_deep_fusion.py`: already implemented CPU deep fusion training primitives.
- `src/models/transformer/train_transformer_fusion.py`: already implemented CPU transformer-style fusion primitives.
- `src/models/deep/train_deep_fusion_gpu.py`: reusable GPU deep fusion backend.
- `src/models/transformer/train_transformer_fusion_gpu.py`: reusable GPU transformer backend.
- `src/evaluation/ablation_runner.py`: older all-in-one CS3 runner with mixed orchestration and training logic.
- `src/visualization/plot_cs1.py`, `src/visualization/plot_cs2.py`, `src/visualization/plot_cs3.py`: reusable publication-style figure builders.
- `tests/emotion_log_labeled.csv`: preserved visual baseline label log used for Paper 1 synthetic-data bootstrapping.

### Existing config structure
- `configs/cs1/default.yaml`
- `configs/cs2/default.yaml`
- `configs/cs3/default.yaml`
- `configs/cs3/comparison.yaml`
- `configs/cs3/comparison_gpu.yaml`
- `configs/cs3/gpu_only_long_train.yaml`
- `configs/cs3/single_model/*.yaml`

The repository already contained useful CS3 configs, but they mixed multiple algorithms inside shared files, which made debugging and fair comparison harder.

## What was missing for the final Paper 1 design
- A strict single-model-per-config experiment pattern.
- A shared config loader with validation and per-experiment path resolution.
- Single-experiment output isolation under `outputs/csv/<case>/<experiment>/`.
- A benchmark runner that only orchestrates single-model configs instead of embedding model logic.
- CS1 and CS2 configs split by experiment instead of one default catch-all file.
- Reusable config templates for new experiments and new benchmarks.
- Tests for config loading, benchmark execution, and result merging.

## Refactor decision

### Keep
- Working training primitives for classical, deep, transformer, and GPU variants.
- The preserved repo baseline visual benchmark.
- The synthetic Paper 1 dataset generation path for placeholder benchmarking.
- Existing plotting utilities and summary-table exporters where still useful.

### Refactor
- Move from shared mixed configs to one-config-per-experiment for CS1, CS2, and CS3.
- Change CS1 and CS2 runners to write per-experiment outputs and config snapshots.
- Change CS3 runners so each script can execute exactly one model config directly.
- Add `src/common/config_loader.py` and experiment-path helpers.

### Deprecate but preserve
- `src/evaluation/ablation_runner.py` remains in the repo as a legacy compatibility path.
- Paper 1 should now prefer:
  - direct single-experiment runners
  - `src/evaluation/benchmark_runner.py` for aggregation

### Add
- `configs/templates/model_experiment_template.yaml`
- `configs/templates/benchmark_template.yaml`
- CS1 split configs: `simulator_only`, `control_loop`, `playback`, `fault_injection`
- CS2 split configs: `video_audio`, `video_audio_robotstate`, `missing_audio`, `missing_video`
- CS3 split configs for baseline, classical, deep, and transformer models
- `configs/cs3/benchmark_all.yaml`
- new Paper 1 run docs and config catalog

## Why this design is better for Paper 1
- Single-model-per-config makes every experiment reproducible and inspectable.
- Debugging becomes easier because model choice, modalities, and outputs are no longer hidden in one large YAML file.
- Scientific fairness improves because each run has an explicit and frozen experimental definition.
- Benchmark orchestration stays thin and auditable because it only launches pre-defined single experiments and merges their outputs.
