# CS7: KG Plus LLM Explainability Quality and Faithfulness Analysis

## Purpose
Assess whether graph-grounded explanations can improve interpretability without sacrificing factual faithfulness.

## Dataset, Simulator, and Input Assumptions
- Knowledge graph content is synthetic or curated from controlled templates in this phase.
- Prediction bundles come from simulated CS4 and CS5 scenarios.
- Human evaluation can be approximated with structured rubric scoring before live review.

## Method
- Retrieve graph facts relevant to the current alert or recommendation.
- Condition an LLM explanation generator on those facts.
- Compare graph-grounded explanations against ungrounded LLM text and template-only baselines.

## Baselines
- A6: LLM-enabled socially assistive robot dialogue system
- A7: Explainable KG and HITL healthcare robot
- A8: Proposed integrated system

## Metrics
- faithfulness
- citation coverage
- contradiction rate
- clinician usefulness score
- time-to-understanding

## Expected Findings
- KG grounding should improve provenance and reduce unsupported explanations compared with free-form generation.

## Failure Modes
- hallucinated explanations
- incomplete graph coverage
- overly rigid templates
- mismatch between explanation detail and caregiver needs

## Journal-Quality Figure Plan
- explanation quality comparison
- provenance coverage bars
- faithfulness versus usefulness scatter
