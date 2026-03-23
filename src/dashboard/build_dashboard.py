from __future__ import annotations

from pathlib import Path

import pandas as pd


def _table_html(df: pd.DataFrame) -> str:
    return df.to_html(index=False, classes="dataframe", border=0)


def build_dashboard(project_root: Path) -> Path:
    outputs = project_root / "outputs"
    dashboard_dir = outputs / "dashboard"
    dashboard_dir.mkdir(parents=True, exist_ok=True)

    benchmark = pd.read_csv(outputs / "tables" / "benchmark_summary.csv")
    alerts = pd.read_csv(outputs / "csv" / "dashboard_alerts.csv")
    explanations = pd.read_csv(outputs / "csv" / "explanation_examples.csv")
    pilot = pd.read_csv(outputs / "tables" / "pilot_validation_readiness.csv")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Cognitive Caregiving Robot Dashboard</title>
  <style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 32px; background: #f6f8fb; color: #1f2933; }}
    h1, h2 {{ color: #133c55; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 24px; }}
    .card {{ background: white; border-radius: 16px; padding: 22px; box-shadow: 0 10px 24px rgba(19,60,85,.08); }}
    .hero {{ background: linear-gradient(135deg, #133c55, #2a9d8f); color: white; }}
    .hero h1 {{ color: white; margin-top: 0; }}
    .dataframe {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    .dataframe th, .dataframe td {{ padding: 8px 10px; border-bottom: 1px solid #e6edf4; text-align: left; }}
    .links a {{ display: inline-block; margin-right: 14px; color: #2a6f97; text-decoration: none; font-weight: 600; }}
  </style>
</head>
<body>
  <section class="card hero">
    <h1>Human-in-the-Loop Caregiving Dashboard</h1>
    <p>Research prototype dashboard for reviewing risk alerts, adherence issues, explainability outputs, and pilot-readiness status.</p>
    <div class="links">
      <a href="../figures/system_architecture_overview.pdf">Architecture PDF</a>
      <a href="../figures/literature_gap_heatmap.pdf">Literature Gap PDF</a>
      <a href="../figures/case_study_summary_dashboard.pdf">Case Study PDF</a>
    </div>
  </section>
  <div class="grid">
    <section class="card">
      <h2>Benchmark Summary</h2>
      {_table_html(benchmark)}
    </section>
    <section class="card">
      <h2>Recent Alerts</h2>
      {_table_html(alerts.head(8))}
    </section>
    <section class="card">
      <h2>Explanation Examples</h2>
      {_table_html(explanations)}
    </section>
    <section class="card">
      <h2>Pilot Validation Readiness</h2>
      {_table_html(pilot)}
    </section>
  </div>
</body>
</html>
"""
    output_path = dashboard_dir / "index.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path
