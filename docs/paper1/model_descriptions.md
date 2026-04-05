# Model Descriptions for Paper 1 CS3

This file documents the actual models used in the current `CS3` emotion-recognition benchmark implementation. The goal is to make the benchmark transparent and publication-ready by describing the real code path rather than only listing model family names.

## Label Space
The current implemented label space is:
- `happy`
- `sad`
- `neutral`
- `fear`

This is the real label support available in the preserved baseline evidence. The originally planned `angry` class is not yet supported by the preserved benchmark log and is therefore not claimed here.

## Input Modalities
The current CS3 benchmark uses three aligned modality groups:
- `video`
- `audio`
- `context`

The multimodal benchmark inputs are currently generated as synchronized synthetic placeholders from the Paper 1 feature pipeline. They are suitable for validating the benchmark infrastructure and comparison workflow, but they are not a substitute for real caregiving interaction data.

## B0: Existing Repository Visual Baseline
`B0` is the preserved baseline already present in the repository.

Actual description:
- input: visual-only
- source: `tests/emotion_log_labeled.csv`
- prediction source: preserved DeepFace-style face-emotion outputs already stored in the repository log
- evaluation mode: direct benchmark of existing predictions against ground-truth labels in the log
- training in the current Paper 1 runner: none

What it represents:
- a real implemented baseline
- the starting point for comparison against newly added multimodal benchmark families

Important limitation:
- it is not a full multimodal model
- it does not use audio or context
- it should be interpreted as the repository's preserved reference model, not as the strongest final system

## B1: Classical ML Family
The classical family uses concatenated handcrafted features from the selected modalities.

Common pipeline:
- feature assembly: concatenate selected modality vectors into one feature vector
- optional standardization: applied for algorithms that benefit from scaling
- classifier training: fit on the current training split, then evaluate on the held-out split

Supported classical algorithms:

### `svm`
- implementation: `sklearn.svm.SVC`
- kernel: RBF
- probability estimates: enabled
- class balancing: enabled
- scaling: yes

Why it is included:
- strong shallow baseline for small-to-medium feature spaces
- useful reference against more complex fusion models

### `random_forest`
- implementation: `sklearn.ensemble.RandomForestClassifier`
- role: non-linear tree baseline
- scaling: no

Why it is included:
- robust tabular baseline for multimodal handcrafted features
- useful when feature interactions are non-linear

### `logistic_regression`
- implementation: `sklearn.linear_model.LogisticRegression`
- role: linear multiclass baseline
- scaling: yes

Why it is included:
- interpretable shallow baseline
- useful for understanding whether the benchmark is linearly separable

### `extra_trees`
- implementation: `sklearn.ensemble.ExtraTreesClassifier`
- role: randomized tree ensemble baseline
- scaling: no

Why it is included:
- offers a stronger tree-ensemble comparison with greater randomization than standard random forest

### `gradient_boosting`
- implementation: `sklearn.ensemble.GradientBoostingClassifier`
- role: boosting-based classical baseline
- scaling: no

Why it is included:
- useful to compare additive boosting behavior against bagging and kernel methods

## B2: Deep Late-Fusion Family
The deep family is implemented as a late-fusion MLP benchmark.

Common pipeline:
- concatenate selected modalities into one joint feature vector
- standardize the fused vector
- train an `sklearn.neural_network.MLPClassifier`
- use `partial_fit` with `max_iter=1` and `warm_start=True` so the code can track training epoch by epoch

Shared training behavior:
- activation: `relu`
- optimizer: `adam`
- training style: iterative epoch loop with explicit metric logging
- output tracking: training curve, final metrics, checkpoints, and per-epoch progress tracking

Configured deep variants can differ in:
- hidden layer sizes
- learning rate
- epoch count

Example configured variants:
- `late_fusion_mlp_small`
- `late_fusion_mlp_medium`
- `late_fusion_mlp_wide`

What this family represents:
- a stronger multimodal baseline than shallow concatenation alone
- a practical fusion benchmark without requiring a heavy external deep-learning stack

Important limitation:
- this is still a fused tabular benchmark over prepared features, not an end-to-end raw video/audio network

## B3: Lightweight Transformer Fusion Family
The transformer family is a lightweight multimodal fusion surrogate designed for Paper 1 benchmarking.

Actual architecture:
- each modality is projected into a shared hidden space using a learned random-initialized projection matrix
- the three modality embeddings are treated as tokens
- a mean pooled query vector is formed across tokens
- attention weights are computed between the query and tokens
- a weighted fused representation is produced
- the fused representation is concatenated with the query representation
- a linear online classifier (`sklearn.linear_model.SGDClassifier` with `log_loss`) performs final classification

Key configurable parts:
- hidden dimension
- classifier regularization (`alpha`)
- epoch count

Why this family is included:
- it approximates cross-modal attention behavior in a lightweight and reproducible way
- it creates a transformer-style fusion benchmark without introducing a heavy GPU-only training dependency

Important limitation:
- it is a lightweight attention-inspired benchmark, not a full large-scale transformer trained end to end on raw sequences

## H1: Hybrid Ensemble
The hybrid model is an ensemble over selected benchmark members.

Actual description:
- training: none as a separate learner
- inference rule: majority vote across chosen member models
- typical members: one or more classical models plus one deep variant plus one transformer variant

Why it is included:
- tests whether combining diverse error profiles improves robustness
- gives a direct comparison between single-model and ensemble behavior

Important limitation:
- performance depends on the chosen members
- it adds inference overhead because all selected member models must be evaluated

## Model Selection Logic
The benchmark now exports:
- `model_performance_summary.csv`
- `model_ranking.csv`
- `best_model_summary.csv`
- `family_best_summary.csv`

The best model is selected from the configured comparison using:
- primary metric: usually `macro_f1`
- tie-breakers: typically `weighted_f1`, `accuracy`, then `inference_latency_ms`

This ranking policy is configurable from the YAML files.

## Training and Tracking
Deep and transformer families now support:
- command-line epoch overrides
- configurable checkpoint intervals
- per-epoch progress logs
- model elapsed time
- estimated remaining time per model
- total run elapsed time
- CPU usage
- memory usage
- GPU usage when `nvidia-smi` is available

## Honest Scope Statement
The current CS3 comparison framework is research-grade and executable, but the evidence level must be interpreted carefully:
- `B0` is an implemented real baseline
- `B1`, `B2`, `B3`, and `H1` are currently benchmarked on synthetic placeholder multimodal data
- the benchmark is valid as an infrastructure and comparison scaffold
- final publication claims about superiority must wait for real synchronized multimodal caregiving data
