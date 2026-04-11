# Paper 1 Experiment Design

## Design choice
Paper 1 now uses:
- one config per experiment
- one model per config
- one modality setting per config
- one benchmark runner only for automation and aggregation

This is the preferred design for a Q1-style paper because it separates scientific definition from orchestration.

## Why single-model-per-config was chosen
- Reproducibility: every run is fully specified in one file.
- Debugging: when a result changes, only one model and one modality setting need inspection.
- Fair comparison: each benchmark row maps directly to one experiment artifact directory.
- Extensibility: adding a new model later means adding one config and, if needed, one trainer entrypoint.

## Case-study layout

### CS1
- one config per digital-twin mode
- one output directory per mode
- metrics exported as CSV plus config snapshot and summary JSON

### CS2
- one config per modality availability scenario
- one output directory per synchronization setting
- aligned windows, availability tables, and quality summaries exported per experiment

### CS3
- one config per model and modality combination
- one output directory per algorithm condition
- benchmark runner merges only finished single-experiment outputs

## Benchmark layer
`src/evaluation/benchmark_runner.py` is intentionally thin. It:
- reads a list of single-model config files
- dispatches each one to its own trainer
- collects `metrics.csv`
- merges results into `outputs/tables/cs3_master_model_summary.csv`
- derives `outputs/tables/cs3_ablation_summary.csv`
- generates benchmark-level comparison figures

The benchmark runner does not define models internally and does not replace the single-experiment configs.
