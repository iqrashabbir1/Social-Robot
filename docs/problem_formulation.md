# Problem Formulation

## Goal
Design a privacy-aware, explainable, multimodal digital-twin cognitive caregiving robot that supports elderly and homecare settings through:
- affect-aware interaction
- closed-loop health-risk prediction
- medication adherence support and missed-dose reasoning
- anomaly detection
- caregiver oversight and telepresence

## System Inputs
The framework is defined over synchronized multimodal streams.

| Symbol | Description | Example Sources |
| --- | --- | --- |
| `v_t` | visual stream | face camera, pose, mobility camera, RGB-D |
| `a_t` | speech and audio | microphone, VAD, ASR transcript, prosody |
| `p_t` | physiology | ECG, HR, BP, SpO2, respiration, sleep |
| `b_t` | behavior and activity | gait, room transitions, inactivity, routine deviation |
| `m_t` | medication context | schedule, dispenser logs, adherence status |
| `r_t` | robot state | ROS2 topics, navigation state, dialogue state |
| `g_t` | graph context | patient profile, care plan, medication ontology |
| `u_t` | user and caregiver feedback | dashboard labels, overrides, telepresence notes |
| `q_t` | privacy and deployment state | consent mode, edge/cloud route, bandwidth, battery |

## Multimodal State Representation
The synchronized state is:

`s_t = [e_t, h_t, d_t, z_t, q_t]`

where:
- `e_t`: emotion and engagement embedding
- `h_t`: health-risk and physiological summary state
- `d_t`: medication adherence and daily routine state
- `z_t`: digital-twin latent state for patient, robot, and environment
- `q_t`: privacy, compute, and safety context

Candidate factorization:

`s_t = concat(phi_v(v_t), phi_a(a_t), phi_p(p_t), phi_b(b_t), phi_m(m_t), phi_r(r_t), phi_g(g_t), phi_u(u_t), phi_q(q_t))`

## Tasks and Subtasks
### T1. Multimodal Emotion Recognition
- facial affect estimation
- speech emotion recognition
- multimodal fusion
- uncertainty calibration

### T2. Health-Risk Prediction
- fall-risk forecasting
- cardiovascular deterioration warning
- chronic-condition worsening prediction
- patient-specific anomaly detection

### T3. Medication Adherence Reasoning
- reminder scheduling
- delayed or missed-dose classification
- reason attribution such as forgetfulness, refusal, side effects, access issue, or confusion
- caregiver escalation policy

### T4. Reasoning and Explanation
- retrieve graph-grounded evidence
- synthesize explanation text for caregivers
- provide actionable next steps and confidence

### T5. Human Oversight and Telepresence
- dashboard review
- manual override and acknowledgment logging
- telepresence initiation
- cultural adaptation of prompts and interaction style

## Objective Functions
The multi-task objective is:

`min_theta L_total(theta)`

with:
- `w1 * L_emo`
- `w2 * L_risk`
- `w3 * L_adh`
- `w4 * L_anom`
- `w5 * L_align`
- `w6 * L_expl`
- `w7 * L_cal`
- `w8 * L_priv`
- `w9 * L_safe`

### Candidate Task Losses
- `L_emo`: cross-entropy or focal loss for emotion classification
- `L_risk`: binary cross-entropy, survival loss, or time-to-event ranking loss
- `L_adh`: multi-class loss for adherence outcome and reason code
- `L_anom`: reconstruction or contrastive anomaly objective
- `L_align`: multimodal contrastive or cross-modal consistency loss
- `L_expl`: explanation faithfulness and citation coverage penalty
- `L_cal`: expected calibration error surrogate or Brier loss
- `L_priv`: privacy budget or modality-cost penalty
- `L_safe`: penalty for unsafe actions without required human approval

## Constraints
### Safety Constraints
- high-risk recommendations must require dashboard review before actuation
- medication overrides must be logged with provenance
- autonomous actions must stay within policy-defined authority bounds

### Privacy Constraints
- configurable sensing profiles from minimal to rich sensing
- edge-first execution for sensitive media where possible
- audit trails for data access and explanation provenance

### Deployment Constraints
- bounded latency for reminder, risk alert, and telepresence initiation
- robustness to missing modalities
- graceful degradation under bandwidth or sensor failure

### Reproducibility Constraints
- every table and plot must map back to CSV sources
- all simulated outputs must be labeled as simulated
- no fabricated empirical performance

## Hierarchical Decision Flow
1. Acquire and time-align modalities through ROS2 or ingestion adapters.
2. Estimate low-level features and embeddings.
3. Produce task-level predictions for emotion, risk, adherence, and anomalies.
4. Update digital-twin patient and environment state.
5. Query the knowledge graph for contextual grounding.
6. Generate explanation candidates with confidence and provenance.
7. Route outcome to robot dialogue, caregiver dashboard, telepresence, or escalation channel.
8. Apply privacy and safety gates before any high-stakes action.
9. Log decisions, human feedback, and system uncertainty for retrospective evaluation.

## Module Interfaces
### Sensor Interface
Input:
- raw modality streams

Output:
- timestamped observation packets

### Synchronization Interface
Input:
- observation packets

Output:
- synchronized multimodal window

### Prediction Interface
Input:
- multimodal window plus digital-twin context

Output:
- task predictions with confidence and uncertainty

### Explanation Interface
Input:
- predictions, patient graph, rule context, and dashboard policy

Output:
- evidence-linked explanation bundle

### Oversight Interface
Input:
- prediction and explanation bundle

Output:
- approval, rejection, annotation, telepresence action, or escalation log

## Evaluation Metrics
### MER
- accuracy
- macro F1
- balanced accuracy
- AUROC where applicable
- calibration error
- latency per inference

### Health-Risk Prediction
- AUROC
- AUPRC
- sensitivity at fixed false-alert rate
- time-to-warning
- calibration error

### Medication Adherence
- reminder delivery rate
- adherence classification F1
- missed-dose reason accuracy or macro F1
- escalation precision
- caregiver burden metrics

### Explainability
- factual faithfulness
- evidence citation coverage
- clinician usefulness rating
- contradiction rate
- time-to-understanding

### HITL and Deployment
- override rate
- alert acknowledgment latency
- false escalation rate
- edge latency
- memory and compute footprint
- privacy-utility tradeoff index

## Evidence Separation Rule
Every experiment, table, or figure should be tagged as one of:
- `implemented_real_baseline`
- `simulation_based_evaluation`
- `planned_experiment`

This rule is enforced throughout the generated workspace.
