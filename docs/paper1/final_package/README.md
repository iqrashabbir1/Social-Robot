# Paper 1 Final Package

This package is the clean, single-folder version of Paper 1 for writing, review, and supervision.

It does not replace the rest of `docs/paper1`. Instead, it organizes the current stable story in one place while keeping all previous examples and notes intact.

## What this package contains

- `paper1_analysis.md`
  - full case-study analysis
  - what was validated
  - what remains preliminary

- `algorithms_and_comparison.md`
  - all algorithms used in the controlled benchmark
  - training and testing comparison
  - best validation model vs best external-test model

- `detailed_approach.md`
  - exact technical workflow
  - datasets
  - training strategy
  - runtime strategy
  - monitoring and evaluation flow

- `artifact_index.md`
  - the tables and figures to use in the paper
  - which artifacts are primary vs secondary

- `digital_twin_predictive_validation.md`
  - manuscript-ready update for the upgraded digital twin
  - Section 3.3 equation wording
  - Table 2 synchronization row
  - new Section 4.10 predictive-validation text

- `benchmark_1000_epoch/`
  - dedicated clarification folder for the long comparison run
  - explains the algorithm families, the 1000-epoch policy, and monitoring

## Short Paper 1 summary

Paper 1 should be framed as:
- simulation-first
- ROS 2 compatible
- technically validated
- preliminary
- strengthened by controlled dataset evaluation and ROS replay

The core evidence is now split into two complementary layers:

1. Runtime integration evidence
- ROS 2 live, playback-grounded, and hybrid runtime validation

2. Controlled perception evidence
- public-dataset training, held-out validation, and cross-dataset testing

## Recommended reading order

1. `paper1_analysis.md`
2. `algorithms_and_comparison.md`
3. `detailed_approach.md`
4. `artifact_index.md`
5. `benchmark_1000_epoch/README.md`
## Latest Manuscript Companion

For the easiest Section 4-6 review with the generated plots embedded in context, start here:

- `docs/paper1/final_package/manuscript_sections_4_6_with_plots.md`
