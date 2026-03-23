# CS1: ROS2 Plus Digital Twin Validation

## Purpose
Validate the digital-twin orchestration concept, ROS2 topic decomposition, and replayable safety-aware scenario execution before real hardware deployment.

## Dataset, Simulator, and Input Assumptions
- ROS2 topics are simulated or replayed from logs.
- Patient state is represented through synthetic risk, adherence, and emotion updates.
- Existing baseline MER outputs can be injected as a perception source.

## Method
- Build ROS2-aligned topic schema for sensing, risk alerts, dashboard events, and telepresence state.
- Update a patient-centered digital twin at each synchronized step.
- Evaluate whether alerts, override requirements, and state transitions remain internally consistent.

## Baselines
- A3: ROS2 or digital-twin robot without full cognition stack
- A8: Proposed integrated system

## Metrics
- state transition consistency
- event latency per topic
- replay determinism
- escalation trigger correctness
- override handling latency

## Expected Findings
- The digital twin should improve traceability of system decisions and scenario reproducibility.
- The integrated system should expose more clinically meaningful state than a simulation-only robot baseline.

## Failure Modes
- topic desynchronization
- stale twin state
- inconsistent alert escalation
- missing provenance for override actions

## Journal-Quality Figure Plan
- architecture-aware scenario flow
- topic latency summary
- twin state transition coverage
