# Reproducibility

This repository generates the PAEMDT Paper 1 technical and experimental artifacts from local scripts. The current evidence is technical/experimental and should not be interpreted as clinical validation.

## Environment

- Recommended Python: 3.10 or 3.11.
- Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Core Commands

```bash
python -m src.evaluation.run_benchmarks --project-root .
python -m src.evaluation.run_domain_adaptation --project-root .
python -m src.evaluation.run_repeated_cv_statistics --project-root .
python -m src.evaluation.run_dp_privacy_accounting --project-root .
python -m src.visualization.generate_all_figures --project-root .
python -m src.dashboard.build_dashboard --project-root .
python -m src.orchestration.run_full_local_pipeline --project-root .
```

## Expected Output Folders

- `outputs/csv/`
- `outputs/tables/`
- `outputs/figures/`
- `experiments/results/paper_tables/`
- `docs/paper1/final_package/`

## Expected CSV Files

- `outputs/csv/domain_generalization_results.csv`
- `outputs/csv/dp_privacy_accounting.csv`
- `outputs/csv/repeated_cv_results.csv`
- `outputs/csv/calibration_results.csv`
- `outputs/csv/missing_modality_results.csv`
- `outputs/csv/privacy_latency_results.csv`
- `outputs/csv/digital_twin_sync_results.csv`
- `outputs/csv/evidence_maturity_matrix.csv`

## Expected Table Files

- `outputs/tables/enhanced_benchmark_comparison.csv`
- `outputs/tables/domain_adaptation_progression.csv`
- `outputs/tables/repeated_cv_summary.csv`
- `outputs/tables/statistical_test_summary.csv`
- `outputs/tables/privacy_accounting_summary.csv`
- `outputs/tables/calibration_summary.csv`
- `outputs/tables/missing_modality_summary.csv`
- `outputs/tables/privacy_latency_summary.csv`
- `outputs/tables/digital_twin_sync_summary.csv`
- `outputs/tables/evidence_maturity_summary.csv`

## Expected Figure Files

Figures 3-10 are regenerated as PNG, PDF, and SVG under `outputs/figures/`:

- `Figure_3_Domain_Generalization_Gap.*`
- `Figure_4_Robustness_Ratio.*`
- `Figure_5_Ablation_Analysis.*`
- `Figure_6_Repeated_CV_Confidence_Intervals.*`
- `Figure_7_ECE_Comparison.*`
- `Figure_8_Missing_Modality_Robustness.*`
- `Figure_9_Privacy_Utility_Latency.*`
- `Figure_10_Evidence_Maturity_Dashboard.*`

## Evidence Boundary

The current repository supports implemented real baselines, benchmark-supported enhanced experiments, simulation-supported robustness and digital-twin analyses, and prototype dashboard artifacts. It does not yet provide clinical deployment, ethics approval, live wearable integration, assisted-living pilot execution, or clinician-validated prospective outcomes.
