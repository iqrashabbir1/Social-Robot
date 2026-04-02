# CS2: Multimodal Sensing and Synchronization

## Goal
Create a clean aligned multimodal data pipeline for later robot benchmarking.

## Modalities
- video
- audio
- robot context
- optional physiology placeholder

## Pipeline
- timestamp alignment with tolerance
- modality availability tracking
- missing-modality handling
- fixed-window construction with overlap

## Current Outputs
- `outputs/csv/cs2/session_metadata.csv`
- `outputs/csv/cs2/window_index.csv`
- `outputs/csv/cs2/modality_availability.csv`
- `outputs/csv/cs2/sync_quality_metrics.csv`
- `outputs/figures/cs2/modality_availability_heatmap.png`
- `outputs/figures/cs2/synchronization_quality_comparison.png`
- `outputs/figures/cs2/missing_modality_robustness.png`

## Current Run Snapshot
- aligned nominal mean alignment error: `52.6608 ms`
- aligned nominal full-modality window rate: `0.5072`
- missing-modality stress mean alignment error: `63.1930 ms`
- missing-modality stress full-modality window rate: `0.4159`

## Interpretation
The current run validates the synchronization code path and export logic. It does not yet represent a real robot sensor suite or site-collected multimodal caregiving dataset.

## TODO for Real Integration
- connect real ROS2 image and audio streams
- log robot state from simulator or hardware
- add real physiology streams if available
- repeat the same windowing and synchronization study on collected sessions
