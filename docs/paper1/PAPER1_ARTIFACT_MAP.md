# Paper 1 Artifact Map

This map links manuscript artifacts for PAEMDT to repository scripts, source CSV files, output paths, evidence level, and paper section. The evidence boundary is intentionally explicit: these artifacts support technical and experimental validation, not clinical deployment validation.

| Artifact | Manuscript item | Source CSV/table | Generating script | Output path | Evidence level | Section |
|---|---|---|---|---|---|---|
| Figure 3 | Domain-generalization gap | `outputs/csv/domain_generalization_results.csv` | `src/visualization/plot_domain_generalization.py` | `outputs/figures/Figure_3_Domain_Generalization_Gap.png`; `.pdf`; `.svg` | benchmark-supported experimental module | 4.1 |
| Figure 4 | Robustness ratio | `outputs/csv/domain_generalization_results.csv` | `src/visualization/plot_domain_generalization.py` | `outputs/figures/Figure_4_Robustness_Ratio.png`; `.pdf`; `.svg` | benchmark-supported experimental module | 4.1 |
| Figure 5 | Ablation analysis | `outputs/tables/ablation_summary.csv`; `outputs/tables/paper1_table_ablation_summary.csv` | `src/visualization/plot_ablation_analysis.py` | `outputs/figures/Figure_5_Ablation_Analysis.png`; `.pdf`; `.svg` | benchmark-supported and simulation-supported module evidence | 4.5 |
| Figure 6 | Repeated CV confidence intervals | `outputs/tables/repeated_cv_summary.csv` | `src/visualization/plot_statistical_significance.py` | `outputs/figures/Figure_6_Repeated_CV_Confidence_Intervals.png`; `.pdf`; `.svg` | manuscript-facing statistical uncertainty estimate | 4.6 |
| Figure 7 | Expected calibration error comparison | `outputs/csv/calibration_results.csv` | `src/visualization/plot_calibration_analysis.py` | `outputs/figures/Figure_7_ECE_Comparison.png`; `.pdf`; `.svg` | benchmark-supported calibration analysis | 4.7 |
| Figure 8 | Missing-modality robustness | `outputs/csv/missing_modality_results.csv` | `src/visualization/plot_missing_modality_robustness.py` | `outputs/figures/Figure_8_Missing_Modality_Robustness.png`; `.pdf`; `.svg` | simulation-supported robustness analysis | 4.8 |
| Figure 9 | Privacy-utility-latency trade-off | `outputs/csv/privacy_latency_results.csv` | `src/visualization/plot_privacy_latency_pareto.py` | `outputs/figures/Figure_9_Privacy_Utility_Latency.png`; `.pdf`; `.svg` | deployment-oriented technical benchmark | 4.9 |
| Figure 10 | Evidence maturity dashboard | `outputs/csv/evidence_maturity_matrix.csv` | `src/visualization/plot_evidence_maturity.py` | `outputs/figures/Figure_10_Evidence_Maturity_Dashboard.png`; `.pdf`; `.svg` | evidence-boundary documentation | 4.10 |
| Table 4 | Enhanced benchmark comparison | `outputs/tables/enhanced_benchmark_comparison.csv` | `src/evaluation/run_domain_adaptation.py` | `outputs/tables/enhanced_benchmark_comparison.csv`; `experiments/results/paper_tables/table4_multi_algorithm_benchmark.csv` | benchmark-supported experimental module | 4.1 |
| Table 4b | Domain-adaptation progression | `outputs/tables/domain_adaptation_progression.csv` | `src/evaluation/run_domain_adaptation.py` | `outputs/tables/domain_adaptation_progression.csv`; `experiments/results/paper_tables/table_domain_adaptation_results.csv` | manuscript-facing experimental summary | 4.1 |
| Table 5 | Ablation summary | `outputs/tables/ablation_summary.csv`; `outputs/tables/paper1_table_ablation_summary.csv` | `src/visualization/plot_ablation_analysis.py` | `experiments/results/paper_tables/table5_ablation.csv` | benchmark-supported and simulation-supported module evidence | 4.5 |
| Table 6 | Missing-modality summary | `outputs/tables/missing_modality_summary.csv` | `src/evaluation/run_missing_modality_robustness.py` | `outputs/tables/missing_modality_summary.csv` | simulation-supported robustness analysis | 4.8 |
| Privacy accounting | DP-SGD epsilon and delta | `outputs/csv/dp_privacy_accounting.csv` | `src/evaluation/run_dp_privacy_accounting.py` | `outputs/tables/privacy_accounting_summary.csv` | technical privacy validation, not certification | 4.9 |
| Digital-twin sync | 124.0 +/- 67.0 ms synchronization | `outputs/csv/digital_twin_sync_results.csv` | `src/evaluation/run_digital_twin_sync_analysis.py` | `outputs/tables/digital_twin_sync_summary.csv` | technical synchronization measurement | 3.3, 4.10 |

## Evidence Boundary

The repository distinguishes implemented real baselines, benchmark-supported experimental modules, simulation-supported modules, prototype modules, and planned clinical validation. None of the artifacts above should be described as prospective clinical validation, assisted-living deployment, ethics-approved field testing, or clinician-validated pilot evidence.
