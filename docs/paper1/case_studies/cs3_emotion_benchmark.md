# CS3: Emotion Recognition Benchmark

## Goal
Benchmark baseline, classical, deep-fusion, and transformer-style models for the caregiving-robot setting.

## Benchmark Families
- `B0`: preserved existing visual baseline from the repository
- `B1`: classical multimodal benchmark family including SVM, Random Forest, Logistic Regression, Extra Trees, and optional boosting-style baselines
- `B2`: deep late-fusion MLP family with configurable hidden-layer sizes and learning rates
- `B3`: transformer-style lightweight cross-modal fusion family with configurable hidden dimension and online classifier regularization
- `H1`: optional hybrid ensemble using majority voting across selected classical, deep, and transformer members

## Actual Model Descriptions
Detailed model descriptions are documented in:
- `docs/paper1/model_descriptions.md`

Short implementation summary:
- `B0` evaluates preserved visual predictions already stored in the repository log
- `B1` trains shallow classifiers on concatenated handcrafted video, audio, and context features
- `B2` trains an epoch-tracked late-fusion MLP using `MLPClassifier` with `partial_fit`
- `B3` trains a lightweight attention-inspired multimodal fusion surrogate using projected modality tokens and an online logistic classifier
- `H1` performs majority-vote fusion across selected member models and is used as a hybrid comparison baseline rather than a separately trained network

## Current Label Space
The repository currently supports the four classes:
- happy
- sad
- neutral
- fear

This reflects the actual preserved baseline evidence in `tests/emotion_log_labeled.csv`. The originally intended `angry` class remains a future data-collection target.

## Current Outputs
- `outputs/csv/cs3/model_performance_summary.csv`
- `outputs/csv/cs3/confusion_matrix_baseline.csv`
- `outputs/csv/cs3/confusion_matrix_deep.csv`
- `outputs/csv/cs3/confusion_matrix_transformer.csv`
- `outputs/csv/cs3/ablation_results.csv`
- `outputs/csv/cs3/training_curves.csv`
- `outputs/figures/cs3/model_comparison_barplot.png`
- `outputs/figures/cs3/confusion_matrix_baseline.png`
- `outputs/figures/cs3/confusion_matrix_deep.png`
- `outputs/figures/cs3/confusion_matrix_transformer.png`
- `outputs/figures/cs3/ablation_comparison.png`
- `outputs/figures/cs3/training_curves.png`

## Current Run Snapshot
- B0 accuracy: `0.8350`, macro F1: `0.8155`
- B1 accuracy: `1.0000`, macro F1: `1.0000`
- B2 accuracy: `0.9417`, macro F1: `0.7929`
- B3 accuracy: `0.9583`, macro F1: `0.9647`

## Interpretation
- B0 is the preserved implemented real baseline.
- B1, B2, B3, and H1 are synthetic placeholder multimodal benchmarks used to validate the experiment and ablation pipeline.

## TODO for Real Completion
- collect aligned multimodal emotion data for the target caregiving setting
- add real multimodal labels across video, audio, and robot context
- re-run B1 through B3 on real synchronized windows
