# System Architecture

## Summary
The proposed framework upgrades the repository from a baseline multimodal emotion demo into a research-grade cognitive caregiving architecture.

The system is intentionally modular so each component can be benchmarked independently and then composed into an end-to-end caregiving loop.

## Architectural Principles
1. Preserve the current DeepFace plus speech baseline as a reproducible starting point.
2. Decouple sensing, prediction, reasoning, deployment, and oversight.
3. Treat ROS2 and the digital twin as evaluation infrastructure, not just a simulation extra.
4. Keep privacy, explainability, and HITL governance inside the main system loop.
5. Export all analysis artifacts in CSV-backed form for publication workflows.

## Main Modules
### M1. Sensor Adapters
- visual ingestion
- audio ingestion
- physiological ingestion
- wearable and IoT ingestion
- telepresence and dashboard event ingestion

### M2. Synchronization Layer
- timestamp alignment
- missing-modality handling
- windowing
- quality flags

### M3. Representation Layer
- classical feature extraction for baseline studies
- sequential encoders for temporal streams
- multimodal embedding fusion

### M4. Task Heads
- baseline MER
- deep multimodal MER
- health-risk prediction
- anomaly detection
- medication adherence reasoning

### M5. Digital Twin
- patient state mirror
- environment and routine context
- robot state and action history
- scenario replay hooks

### M6. Explainability
- knowledge graph
- rule templates
- LLM explanation generator
- provenance and confidence packager

### M7. Oversight and Interaction
- caregiver dashboard
- escalation and override manager
- telepresence controller
- culturally adaptive dialogue policy

### M8. Deployment and Privacy
- edge versus cloud routing
- privacy-aware modality gating
- audit logging
- resource monitoring

## Data Flow
1. Sensors publish timestamped observations.
2. The synchronization service assembles aligned windows and quality tags.
3. Encoders generate modality-specific representations.
4. Prediction heads emit risk, emotion, anomaly, and adherence outputs.
5. The digital twin updates patient, environment, and robot state.
6. The graph plus LLM explanation stack builds evidence-linked summaries.
7. The dashboard and telepresence modules present actionable outputs.
8. Safety and privacy gates filter actions and log any human intervention.

## Preserved Baseline Path
The existing baseline path remains:

`camera -> DeepFace emotion -> speech SVM -> rule fusion -> response rule -> TTS`

This is wrapped as the baseline MER module and used for:
- implemented baseline evidence
- preliminary benchmark
- ablation reference against improved multimodal models

## Proposed Advanced Path
The advanced path adds:

`multimodal sensing -> synchronized state -> cross-modal transformer -> risk and adherence heads -> digital twin -> KG retrieval -> LLM explanation -> HITL dashboard -> telepresence or robot action`

## Research Layers
### Layer 1
Sensing and synchronization

### Layer 2
Features, embeddings, and alignment

### Layer 3
Task-specific prediction

### Layer 4
Reasoning and explanation

### Layer 5
Human oversight, telepresence, privacy, and deployment

## ROS2 and Digital Twin Role
ROS2 supports:
- modular communication
- replayable evaluation
- digital-twin state transitions
- hardware-in-the-loop migration later

The digital twin stores:
- patient latent health state
- routine and medication schedule context
- robot interaction history
- safety status
- telepresence session state

## Human Oversight Logic
Low-risk outputs:
- can be surfaced directly to the robot interaction layer

Medium-risk outputs:
- should be shown to caregivers with recommended follow-up

High-risk outputs:
- require explicit acknowledgment, override, or telepresence escalation

## Artifact Map
The architecture-to-artifact mapping is exported in:
- `outputs/csv/system_module_map.csv`
- `outputs/figures/system_architecture_overview.svg`

Additional figure formats can be regenerated from the Python plotting pipeline once a standard Python runtime is available in the shell.
