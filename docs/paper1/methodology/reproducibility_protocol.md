# Reproducibility Protocol

## Seeds
- CS1 seed: `42`
- CS2 seed: `42`
- CS3 seed: `42`

## Config Files
- `configs/cs1/default.yaml`
- `configs/cs2/default.yaml`
- `configs/cs3/default.yaml`

## Execution Model
All Paper 1 runs are executed through explicit module entrypoints and write CSV outputs before plotting. Figures are generated only from CSV files.

## Evidence Labels
- `implemented_real_baseline`
- `synthetic_placeholder_benchmark`
- `simulation_based_evaluation`

## Current Realism Boundaries
- the preserved visual baseline is real repository evidence
- the digital-twin experiments are software-equivalent simulations
- the synchronized multimodal and B1–B3 benchmark data are current synthetic placeholders
