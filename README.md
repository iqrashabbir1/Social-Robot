# Social Robot

Privacy-aware, explainable, multimodal digital-twin cognitive caregiving robot for elderly and homecare settings.

## What This Repository Contains
This repository upgrades a baseline social-robot prototype into a publication-oriented research workspace centered on:
- multimodal monitoring
- multimodal emotion recognition
- health-risk prediction
- medication adherence reasoning
- ROS2 and digital-twin integration
- knowledge-graph plus LLM explainability
- human-in-the-loop oversight
- privacy-aware and edge-aware deployment
- telepresence and cultural adaptation

## Current Status
### Implemented and Pushed
- preserved baseline face-plus-speech emotion pipeline
- modular research workspace under `docs/`, `src/`, `configs/`, and `outputs/`
- research master plan, problem formulation, and system architecture documents
- literature comparison package
- 8 case-study designs with CSV-backed metrics and summary tables
- reusable evaluation and visualization scaffolding
- publication-style PNG and SVG figures

### Not Yet Completed
- full experimental execution for the advanced multimodal models
- real physiological integration
- populated knowledge graph and dashboard implementation
- pilot deployment validation
- PDF figure export in the current shell setup

No experimental performance was fabricated. The repository separates:
- implemented real baseline
- simulation-based evaluation
- planned experiments

## Key Documents
- [Research master plan](docs/research_master_plan.md)
- [Problem formulation](docs/problem_formulation.md)
- [System architecture](docs/system_architecture.md)
- [Next actions](docs/next_actions.md)

## Case Studies
- [CS1: ROS2 plus digital twin validation](docs/case_studies/case_study_1.md)
- [CS2: Multimodal sensing and synchronization](docs/case_studies/case_study_2.md)
- [CS3: MER benchmark](docs/case_studies/case_study_3.md)
- [CS4: Health-risk prediction and anomaly detection](docs/case_studies/case_study_4.md)
- [CS5: Medication adherence reasoning](docs/case_studies/case_study_5.md)
- [CS6: HITL dashboard and override safety](docs/case_studies/case_study_6.md)
- [CS7: KG plus LLM explainability](docs/case_studies/case_study_7.md)
- [CS8: Privacy, edge, telepresence integrated scenario](docs/case_studies/case_study_8.md)

## Important Outputs
### Tables and CSV
- `outputs/tables/literature_comparison_matrix.csv`
- `outputs/tables/literature_comparison_summary.md`
- `outputs/csv/system_module_map.csv`
- `outputs/csv/case_study_*_metrics.csv`
- `outputs/tables/case_study_*_summary.csv`

### Figures
- `outputs/figures/literature_gap_heatmap.png`
- `outputs/figures/literature_radar_or_bar_comparison.png`
- `outputs/figures/system_architecture_overview.png`
- `outputs/figures/case_study_summary_dashboard.png`

## Repository Structure
```text
docs/                research reports and case studies
src/                 models, pipelines, evaluation, visualization
configs/             benchmark and figure-generation config
outputs/             csv, figures, tables, logs
perception/          preserved baseline perception modules
fusion/              preserved baseline fusion logic
response/            preserved baseline response and TTS logic
tests/               baseline scripts and research smoke tests
data/                speech datasets and baseline models
```

## Baseline That Was Preserved
The existing webcam and DeepFace-style 4-class emotion pipeline is preserved as the baseline MER reference, along with speech emotion and rule-based fusion components.

## Recommended Next Step
Expose a standard Python runtime in the shell and then run the benchmark and plotting pipeline from:
- `src/evaluation/run_benchmarks.py`
- `src/visualization/generate_all_figures.py`
