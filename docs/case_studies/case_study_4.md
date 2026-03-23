# CS4: Health-Risk Prediction and Anomaly Detection

## Purpose
Define a rigorous evaluation path for proactive risk prediction and anomaly detection without overclaiming unavailable physiological experiments.

## Dataset, Simulator, and Input Assumptions
- Physiology and mobility streams are currently simulated.
- Risk scenarios include fall precursors, cardiovascular deviation, and chronic-condition worsening.
- Patient baselines are individualized through the digital-twin state.

## Method
- Compare shallow physiological risk models against sequence models and the full integrated stack.
- Add anomaly detection for patient-specific deviations.
- Use simulation-backed trajectories to estimate time-to-warning and false-alert burden.

## Baselines
- A1: Classical wearable monitoring plus shallow ML risk prediction
- A4: IoT-edge healthcare monitoring
- A8: Proposed integrated system

## Metrics
- AUROC
- AUPRC
- sensitivity at fixed false-alert rate
- time-to-warning
- anomaly detection delay
- calibration

## Expected Findings
- The integrated stack should provide better context sensitivity than purely physiology-first baselines because it can condition on behavior, adherence, and affect.

## Failure Modes
- false alerts from noisy vitals
- weak personalization
- simulation assumptions that do not transfer to real deployments
- uncalibrated risk outputs

## Journal-Quality Figure Plan
- ROC and PR curves
- calibration plot
- warning lead-time violin or bar chart
- anomaly timeline figure
