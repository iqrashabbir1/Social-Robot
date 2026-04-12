# Detailed Approach

## 1. Overall Paper 1 strategy

Paper 1 now follows a layered methodology:

1. Runtime and architecture validation
2. Dataset-based controlled perception evaluation
3. ROS dataset replay for pipeline validation
4. Multi-algorithm comparison
5. Cross-dataset testing for external validity

This is more realistic and practical than relying on random live snapshots as the main evidence.

## 2. Runtime layer

The project keeps multiple runtime modes intact:
- `ros2_playback_grounded`
- `ros2_live_laptop_sensors`
- `ros2_live_windows_stream_wsl_core`
- `ros2_dataset_replay`

These modes serve different evidence roles:
- live and hybrid modes validate runtime integration
- playback and dataset replay validate reproducible data flow
- offline dataset evaluation validates perception quality under controlled conditions

## 3. Dataset layer

### Base training dataset
- `RAVDESS`
- used for supervised training
- held-out `20%` used for local validation

### External test dataset
- `CREMA-D`
- used only after training for cross-dataset testing

### Planned additional public dataset
- `EmoryNLP`
- support has been prepared for a later extension

## 4. Model-training procedure

### Long-run single-model training

The long monitored training path uses:
- `src.models.vision.train_image_emotion_classifier`
- `src.evaluation.run_cross_dataset_generalization`

Key settings used in the current long run:
- epochs: `1000`
- batch size: `64`
- device: `cuda`

Monitoring added for long runs:
- per-epoch console logs
- per-step console logs
- loss tracking
- latest status JSON
- training progress CSV
- event log CSV

### Multi-algorithm comparison

The comparison workflow uses:
- `src.evaluation.run_multialgorithm_emotion_case_study`

This workflow:
1. trains all algorithms on the same `RAVDESS` train split
2. validates each algorithm on the same held-out split
3. tests each trained algorithm on `CREMA-D`
4. exports comparison tables

## 5. Why both training and testing views matter

Training metrics alone can be misleading.

Held-out validation tells us:
- how well the model fits the source domain
- whether the training procedure is stable

External testing tells us:
- whether the model generalizes outside the source dataset
- whether the apparent winner remains strong under distribution shift

This is why Paper 1 now compares:
- train metrics
- held-out validation metrics
- external-test metrics

## 6. ROS dataset replay path

The project also includes:
- `social_robot/dataset_replay_node.py`
- `launch/dataset_replay.launch.py`

This allows dataset images or frames to be replayed through:
- `/camera/image_raw`

That means the same ROS pipeline can be exercised with controlled inputs rather than only with live camera data.

This is important because it connects:
- controlled dataset evidence
- live/runtime system evidence

## 7. Practical experimental flow

The practical Paper 1 flow is:

1. Download and prepare `RAVDESS`
2. Download and prepare `CREMA-D`
3. Train on `RAVDESS`
4. Validate on held-out `RAVDESS`
5. Test on `CREMA-D`
6. Compare multiple algorithms
7. Use ROS dataset replay to demonstrate integration through the pipeline

## 8. Best use in the manuscript

The paper should use:
- architecture and runtime figures for systems validation
- dataset tables for controlled model comparison
- cross-dataset results for external validity discussion

This makes the paper stronger because it separates:
- runtime feasibility
- benchmark accuracy
- robustness under dataset shift

