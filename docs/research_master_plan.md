# Research Master Plan

## Refined Title
Privacy-Aware, Explainable, Multimodal Digital-Twin Cognitive Caregiving Robot for Elderly and Homecare Support

## Positioning
This project is framed as a privacy-aware, explainable, multimodal digital-twin cognitive caregiving robot with closed-loop health-risk prediction, medication adherence reasoning, affect-aware interaction, and human oversight.

The existing repository already contains an implemented baseline multimodal emotion-recognition stack:
- webcam/DeepFace face emotion analysis
- speech emotion classification with MFCC features and shallow ML
- rule-based fusion and empathetic response generation

That baseline is preserved as the preliminary MER benchmark and ablation reference, not as the final system target.

## State-of-the-Art Problem Statement
Homecare robots for elderly populations often solve isolated problems: monitoring only physiology, supporting only reminders, modeling only social dialogue, or benchmarking only perception. This fragmentation limits clinical usefulness, safety, and publishability because real caregiving workflows require perception, prediction, reasoning, explanation, privacy controls, and human review in the same loop.

The target problem is to design and evaluate a cognitive caregiving robot that:
- synchronizes multimodal sensing from speech, movement, behavioral patterns, and physiological streams
- maintains a ROS2-connected digital twin of patient state, environment context, and robot intent
- predicts health risks, anomalies, and medication non-adherence before they escalate
- adapts interaction based on affect, culture, and telepresence needs
- produces explanations grounded in a knowledge graph and surfaced through an LLM layer
- enforces privacy-aware sensing, edge deployment constraints, and human-in-the-loop safety governance

## Careful Novelty Claims
The following are proposed contributions to be validated, not claimed as achieved results:

1. A unified closed-loop framework that connects multimodal perception, health-risk prediction, medication adherence reasoning, explainability, and caregiver oversight in one architecture.
2. A digital-twin formulation that links ROS2 simulation, patient state estimation, and deployment-aware evaluation.
3. A layered evaluation protocol that separates implemented baseline evidence, simulation-backed evidence, and future real-world pilot plans.
4. A publication-oriented benchmarking package with explicit ablations, qualitative literature comparison, CSV-backed figures, and case-study level validation plans.
5. A privacy-aware deployment view that treats sensing fidelity, latency, explainability, and data minimization as coupled design constraints instead of separate afterthoughts.

## Research Questions
RQ1. How can multimodal sensing and synchronization improve robust estimation of elderly user affect, behavioral context, and care-relevant state under noisy homecare conditions?

RQ2. Can a digital-twin cognitive architecture improve the realism and safety of evaluation for closed-loop caregiving tasks before real-world pilot deployment?

RQ3. How much value is added by moving from single-task models to an integrated stack that jointly supports emotion awareness, health-risk prediction, anomaly detection, medication adherence reasoning, and caregiver intervention?

RQ4. Can knowledge-graph-guided LLM explanations improve interpretability and clinician trust without weakening factual faithfulness or safety governance?

RQ5. What privacy, edge, telepresence, and cultural-adaptation tradeoffs emerge when moving from lab-grade multimodal models toward homecare deployment?

## Target Architecture
The framework is organized into five methodological layers.

### Layer 1: Sensing and Synchronization
- RGB or RGB-D face and posture streams
- speech audio and optional ASR transcripts
- physiological streams such as ECG, HR, BP, SpO2, activity, and mobility
- medication events and adherence logs
- caregiver annotations, override actions, and telepresence session metadata
- ROS2 time synchronization and digital-twin state update hooks

### Layer 2: Feature Extraction and Embeddings
- classical feature extractors for baseline studies
- sequential encoders for speech, vitals, and temporal activity traces
- multimodal embedding alignment for audio, vision, and physiology
- domain adaptation between simulation, public data, and pilot data

### Layer 3: Task-Specific Prediction
- baseline MER model using the current DeepFace plus speech SVM pipeline
- deep multimodal MER models with late fusion
- transformer-based cross-modal reasoning
- health-risk prediction for falls, cardiac deterioration, and chronic-condition worsening
- anomaly detection for deviations from patient-specific baselines
- medication adherence prediction and missed-dose reason tagging

### Layer 4: Reasoning and Explainability
- patient-care knowledge graph
- rule and retrieval layer for evidence grounding
- LLM explanation synthesis bounded by graph evidence and dashboard policy
- confidence, calibration, and uncertainty reporting

### Layer 5: Oversight, Telepresence, and Deployment
- caregiver and clinician dashboard
- human override and escalation policy
- privacy-aware sensing profiles
- edge versus cloud execution routing
- telepresence support and cultural adaptation controls

## Mathematical Framing
At time step `t`, the system observes multimodal inputs:

`x_t = {v_t, a_t, p_t, m_t, b_t, c_t, h_t}`

where:
- `v_t`: visual observations
- `a_t`: audio and speech observations
- `p_t`: physiological observations
- `m_t`: medication events and schedule context
- `b_t`: behavior and mobility context
- `c_t`: environment, culture, telepresence, and deployment context
- `h_t`: human feedback and caregiver intervention signals

The synchronized latent patient-robot state is:

`s_t = f_sync(x_1, ..., x_t, z_twin, k_patient)`

where `z_twin` is the digital-twin state and `k_patient` is patient-specific prior knowledge.

The system jointly optimizes tasks:
- `y_t^emo`: emotion state
- `y_t^risk`: risk of adverse health event
- `y_t^anom`: anomaly state
- `y_t^adh`: adherence outcome and missed-dose reason
- `y_t^act`: recommended robot or caregiver action
- `e_t`: explanation package for human review

Overall training and decision objective:

`L_total = w_emo L_emo + w_risk L_risk + w_anom L_anom + w_adh L_adh + w_align L_align + w_cal L_cal + w_safe L_safe + w_priv L_priv`

subject to:
- latency budgets for edge deployment
- privacy budgets and data minimization rules
- human-override constraints for high-risk actions
- fairness and subgroup robustness considerations
- explainability faithfulness constraints

## Evaluation Strategy
The project explicitly separates three evidence levels.

### Implemented Real Baseline
- current face emotion baseline from webcam and DeepFace
- current speech emotion pipeline based on MFCC and shallow ML
- current rule-based fusion and empathetic response logic

### Simulation-Based Evaluation
- ROS2 and digital-twin case studies
- latency and resource tradeoff scenarios
- privacy-aware routing and telepresence scenarios
- HITL workflow timing and escalation simulations

### Placeholder Experimental Plan
- full multimodal transformer training
- real physiological integration
- pilot deployment in assisted-living or hospital-like settings
- clinician-facing explanation studies

## Case Studies
CS1. ROS2 plus digital twin validation

CS2. Multimodal sensing and data synchronization

CS3. Classical ML versus deep multimodal versus transformer MER benchmark

CS4. Health-risk prediction and anomaly detection

CS5. Medication adherence reasoning and missed-dose justification

CS6. HITL dashboard and override safety analysis

CS7. KG plus LLM explainability quality and faithfulness analysis

CS8. Privacy, edge, telepresence, and integrated end-to-end deployment

## Risks and Mitigations
### Data Availability Risk
- Current repository lacks synchronized physiological and robot-state datasets.
- Mitigation: treat health-risk, adherence, and digital-twin studies as simulation-backed plans until data are collected.

### Overclaiming Risk
- A sophisticated architecture can look more complete than the available evidence.
- Mitigation: every table, CSV, and document labels evidence as implemented baseline, simulation-backed, or planned.

### Domain Shift Risk
- Public affect and speech datasets do not match elderly homecare settings.
- Mitigation: include personalization, domain adaptation, and pilot-data collection in the roadmap.

### Explainability Risk
- LLM explanations may sound persuasive without being faithful.
- Mitigation: force explanation generation to cite graph facts, thresholds, and provenance fields.

### Privacy Risk
- Multimodal sensing increases identifiability.
- Mitigation: edge inference, configurable sensing profiles, consent-aware routing, and privacy-utility reporting.

### Safety Risk
- Closed-loop action selection can be unsafe without clear escalation rules.
- Mitigation: keep high-stakes recommendations advisory-only until validated, with mandatory caregiver review for high-risk decisions.

## Publication Roadmap
### Paper 1
Baseline multimodal emotion benchmark for elderly-facing robotic interaction

### Paper 2
Digital-twin cognitive caregiving robot architecture and case-study validation

### Paper 3
Explainability, privacy, and oversight in cognitive caregiving robots

### Pilot Study Paper
Prospective field validation in assisted-living or hospital-like settings after ethics, hardware, and data readiness

## Assumptions Used in This Upgrade
- Existing baseline code under `perception/`, `fusion/`, and `response/` is preserved as-is.
- `data/` already contains reusable speech emotion assets and datasets.
- `ros2_ws/` and `paper/` exist but are not yet mature enough to serve as the central research scaffold.
- A standard Python runtime is not currently available in the active shell, so the reusable Python framework is created but not executed in this pass.
