# CS6: HITL Dashboard and Override Safety Analysis

## Purpose
Measure whether the system supports safe caregiver oversight, not just automated predictions.

## Dataset, Simulator, and Input Assumptions
- Alert bundles are simulated from CS4 and CS5 outputs.
- Caregiver review actions are represented as synthetic dashboard events.
- Override policies follow severity tiers.

## Method
- Present structured alerts with explanations and confidence.
- Track acknowledgment time, override frequency, and escalation pathways.
- Compare automated recommendation-only flow versus explicit HITL review flow.

## Baselines
- A7: Explainable KG and HITL healthcare robot
- A8: Proposed integrated system

## Metrics
- acknowledgment latency
- override rate
- false escalation rate
- action trace completeness
- caregiver workload proxy

## Expected Findings
- A full HITL loop should improve safety governance and auditability even if it adds some response overhead.

## Failure Modes
- dashboard overload
- unclear explanations
- unlogged overrides
- delayed action in urgent scenarios

## Journal-Quality Figure Plan
- alert funnel
- override heatmap
- safety-governance timing chart
