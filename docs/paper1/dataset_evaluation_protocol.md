# Dataset Evaluation Protocol

## Purpose
Paper 1 now separates:
- dataset-based controlled perception evidence
- dataset replay through the ROS2 pipeline
- live hybrid runtime evidence

This split makes the perception evidence more structured than ad hoc live camera captures.

## Supported inputs
- image folders
- optional video folders
- labels from folder structure
- labels from CSV
- test-only mode when labels are not yet available

## Current local default
- dataset root: `data/pilot/sessions/paper1_anchor_demo/frames`
- current local default is a small image-set anchor for pipeline verification
- this local anchor is sufficient for qualitative panels and replay demonstrations
- a larger labeled emotion dataset should replace it for stronger publication-grade benchmarking

## Recommended public datasets
- `RAF-DB` for the main controlled image-evaluation path
- `CREMA-D` for dataset replay and multimodal discussion
- shared target label set for the upgraded Paper 1 path:
  - `happy`
  - `sad`
  - `neutral`
  - `angry`

## Outputs
- predictions CSV
- metrics summary CSV
- confusion matrix CSV when labels exist
- dataset sample panel
- dataset prediction panel
- dataset replay sequence figure

## Recommended use in Paper 1
- use dataset figures as the main controlled perception layer
- use live hybrid runtime figures/tables as system-integration evidence
- do not present unlabeled local dataset metrics as full benchmark evidence
