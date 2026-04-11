# How To Run Benchmarks

## Primary CS3 benchmark

```powershell
cd "C:\Users\masim\OneDrive\Desktop\Social Robot"
.\.venv\Scripts\Activate.ps1
python -m src.evaluation.benchmark_runner --project-root . --config configs/cs3/benchmark_all.yaml
```

## What the benchmark runner does
- launches each listed single-model config one by one
- reads each experiment `metrics.csv`
- merges results into `outputs/tables/cs3_master_model_summary.csv`
- derives an ablation-style summary in `outputs/tables/cs3_ablation_summary.csv`
- renders benchmark-level figures in `outputs/figures/cs3/`

## Primary benchmark outputs
- `outputs/tables/cs3_master_model_summary.csv`
- `outputs/tables/cs3_ablation_summary.csv`
- `outputs/tables/cs3_benchmark_run_manifest.csv`
- `outputs/figures/cs3/model_comparison_barplot.png`
- `outputs/figures/cs3/ablation_comparison.png`
- `outputs/figures/cs3/inference_latency_comparison.png`

## Generate figures after runs
```powershell
python -m src.visualization.generate_all_figures --project-root .
```

## Export paper summary tables
```powershell
python -m src.evaluation.export_results --project-root .
```
