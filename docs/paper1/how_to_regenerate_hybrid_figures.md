# How To Regenerate Hybrid Figures

## From the Windows project root

```powershell
cd "C:\Users\masim\OneDrive\Desktop\Social Robot"
.\.venv\Scripts\Activate.ps1
python -m src.evaluation.collect_hybrid_runtime_metrics --project-root .
python -m src.evaluation.extract_rosbag_summary --project-root . --rosbag-dir "C:\path\to\paper1_hybrid_runtime"
python -m src.visualization.save_camera_sample_frames --project-root . --frame-dir "C:\path\to\hybrid_exported_frames"
python -m src.evaluation.export_results --project-root .
python -m src.visualization.generate_all_figures --project-root . --rosbag-dir "C:\path\to\paper1_hybrid_runtime" --frame-dir "C:\path\to\hybrid_exported_frames"
```

## Minimum regeneration path without rosbag
If you only copied logger CSVs into `outputs/logs/...`:

```powershell
cd "C:\Users\masim\OneDrive\Desktop\Social Robot"
.\.venv\Scripts\Activate.ps1
python -m src.evaluation.collect_hybrid_runtime_metrics --project-root .
python -m src.evaluation.export_results --project-root .
python -m src.visualization.generate_all_figures --project-root .
```

## Important note
If no hybrid rosbag, event logger CSV, or frame export is present, the plotting layer will generate explicit placeholder outputs rather than fabricated evidence.
