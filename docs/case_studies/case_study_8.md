# CS8: Privacy, Edge, Telepresence, and Integrated End-to-End Scenario

## Purpose
Stress-test the integrated architecture under realistic deployment constraints instead of isolated algorithm benchmarks.

## Dataset, Simulator, and Input Assumptions
- Compute, bandwidth, and privacy modes are simulated.
- Telepresence sessions are modeled as workflow events.
- The baseline MER path can be reused as the implemented perception component.

## Method
- Compare edge-only, hybrid, and cloud-assisted routing under privacy policies.
- Trigger telepresence when risk, uncertainty, or human request exceeds thresholds.
- Measure utility, latency, and privacy tradeoffs in a closed-loop scenario.

## Baselines
- A4: IoT-edge healthcare monitoring
- A6: LLM-enabled socially assistive robot dialogue system
- A8: Proposed integrated system

## Metrics
- end-to-end latency
- edge resource usage
- privacy-utility score
- telepresence initiation delay
- task completion success

## Expected Findings
- The integrated system should be more deployment-realistic because it explicitly couples privacy, routing, explanation, and oversight.

## Failure Modes
- cloud dependence
- privacy policy conflicts
- telepresence handoff failure
- degraded utility under strict sensing limits

## Journal-Quality Figure Plan
- privacy-utility tradeoff plot
- resource and latency comparison
- end-to-end workflow figure
