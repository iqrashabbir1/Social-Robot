# CS5: Medication Adherence Reasoning and Missed-Dose Justification

## Purpose
Evaluate the robot as a closed-loop adherence assistant rather than a simple reminder tool.

## Dataset, Simulator, and Input Assumptions
- Medication schedules and dispenser events are simulated in this phase.
- Dialogue context and behavior cues can be replayed from interaction templates.
- Missed-dose reason labels are defined as a controlled taxonomy.

## Method
- Trigger reminders based on schedule and context.
- Infer adherence status and likely reason for delay or missed dose.
- Escalate to caregiver review when risk or ambiguity crosses policy thresholds.

## Baselines
- A2: Medication-management robot only
- A6: LLM-enabled socially assistive robot dialogue system
- A8: Proposed integrated system

## Metrics
- reminder delivery success
- adherence state F1
- missed-dose reason macro F1
- escalation precision
- caregiver follow-up rate

## Expected Findings
- The integrated system should be stronger than reminder-only systems because it reasons about behavior, affect, and risk before escalation.

## Failure Modes
- spurious reason inference
- reminder fatigue
- excessive escalations
- unsafe automation without review

## Journal-Quality Figure Plan
- adherence workflow figure
- reason distribution chart
- escalation precision and burden chart
