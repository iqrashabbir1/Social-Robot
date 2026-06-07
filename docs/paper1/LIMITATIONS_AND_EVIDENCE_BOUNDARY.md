# Limitations and Evidence Boundary

This document states the evidence boundary for Paper 1 so repository artifacts are not overinterpreted.

## Implemented Real Baseline

The repository preserves implemented baseline emotion-recognition and runtime components. These are real code paths and logs, but they do not by themselves constitute clinical validation.

## Benchmark-Supported Enhanced Experiments

The enhanced PAEMDT manuscript reports RAVDESS source-domain validation, CREMA-D external-domain evaluation, domain adaptation, repeated cross-validation estimates, calibration, and privacy-preserving training. These are benchmark-supported technical experiments.

## Simulation-Supported Modules

Physiology, medication, HITL routing, privacy trade-off analysis, missing-modality robustness, and digital-twin validation include simulation-supported or manuscript-facing outputs where real synchronized caregiving data are not available.

## Prototype Modules

The dashboard, ROS2/digital-twin scaffold, and evidence-maturity view are prototype or technical integration artifacts. They support system design and reproducibility, not clinical deployment claims.

## Not Yet Completed

- No clinical deployment yet.
- No real assisted-living pilot yet.
- No ethics or consent approval yet.
- No clinician-validated prospective trial yet.
- No live wearable or bedside hardware integration yet.
- No claim of clinical safety, efficacy, diagnosis, or treatment performance.

## Appropriate Interpretation

The repository is a reproducibility and validation package for a technical PAEMDT research framework. It is not evidence of clinical efficacy or autonomous clinical deployment.

The repository should be described as a technical and experimental platform for PAEMDT research. Field deployment and clinical validation remain future work.
