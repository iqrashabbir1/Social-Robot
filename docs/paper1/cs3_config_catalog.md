# CS3 Config Catalog

## Baseline
- `configs/cs3/baseline_visual.yaml`
  - preserved repo visual baseline
  - modality: video
  - evidence level: implemented real baseline

## Classical
- `configs/cs3/svm_video.yaml`
  - SVM
  - modality: video
- `configs/cs3/svm_video_audio.yaml`
  - SVM
  - modality: video + audio
- `configs/cs3/rf_video_audio.yaml`
  - Random Forest
  - modality: video + audio
- `configs/cs3/xgboost_video_audio_context.yaml`
  - XGBoost
  - modality: video + audio + context
  - note: requires `xgboost` to be installed separately

## Deep
- `configs/cs3/deep_fusion_video_audio.yaml`
  - late-fusion MLP
  - modality: video + audio
- `configs/cs3/deep_fusion_video_audio_context.yaml`
  - late-fusion MLP
  - modality: video + audio + context

## Transformer
- `configs/cs3/transformer_video_audio.yaml`
  - lightweight fusion transformer
  - modality: video + audio
- `configs/cs3/transformer_video_audio_context.yaml`
  - lightweight fusion transformer
  - modality: video + audio + context

## Benchmark config
- `configs/cs3/benchmark_all.yaml`
  - orchestration file for the primary Paper 1 benchmark
  - launches the baseline, classical, deep, and transformer configs that are currently runnable in the repo

## Adding a new model later
1. Copy `configs/templates/model_experiment_template.yaml`
2. Set `experiment_name`
3. Set `model.family`
4. Set `model.name`
5. Set `modalities.selected`
6. Set training hyperparameters
7. Run the matching trainer directly
8. Add the config path to `configs/cs3/benchmark_all.yaml` if it should enter the aggregate benchmark
