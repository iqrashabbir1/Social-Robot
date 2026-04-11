from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.io_utils import read_yaml, write_dataframe, write_json, write_yaml
from src.common.logging_utils import get_logger
from src.common.paths import Paper1Paths
from src.models.classical.train_classical import run_classical_experiment
from src.models.deep.train_deep_fusion import run_deep_experiment
from src.models.transformer.train_transformer_fusion import run_transformer_experiment
from src.visualization.plot_cs3 import generate_cs3_figures


def _resolve_config_path(project_root: Path, config_path: str) -> Path:
    candidate = Path(config_path)
    return candidate if candidate.is_absolute() else (project_root / candidate).resolve()


def _dispatch_single_config(project_root: Path, config_path: Path) -> dict[str, str]:
    config = read_yaml(config_path)
    model_cfg = config.get("model", {})
    family = str(model_cfg.get("family", "")).strip().lower()
    if family in {"baseline", "classical"}:
        return run_classical_experiment(project_root, config_path)
    if family == "deep":
        return run_deep_experiment(project_root, config_path)
    if family == "transformer":
        return run_transformer_experiment(project_root, config_path)
    raise ValueError(f"Unsupported model family '{family}' in {config_path}.")


def merge_result_tables(metrics_paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in metrics_paths if path.exists()]
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    sort_columns = [column for column in ("macro_f1", "weighted_f1", "accuracy") if column in merged.columns]
    if sort_columns:
        merged = merged.sort_values(sort_columns, ascending=[False] * len(sort_columns)).reset_index(drop=True)
    return merged


def run_cs3_benchmark(project_root: Path, config_path: Path) -> dict[str, str]:
    resolved_root = project_root.resolve()
    benchmark_cfg = read_yaml(config_path.resolve())
    benchmark_name = str(benchmark_cfg.get("benchmark_name", "cs3_benchmark")).strip() or "cs3_benchmark"
    config_files = [str(item).strip() for item in benchmark_cfg.get("config_files", []) if str(item).strip()]
    continue_on_error = bool(benchmark_cfg.get("options", {}).get("continue_on_error", False))
    if not config_files:
        raise ValueError(f"No config_files listed in {config_path}.")

    paths = Paper1Paths.from_project_root(resolved_root)
    paths.ensure()
    logger = get_logger(f"paper1.cs3.benchmark.{benchmark_name}", paths.outputs_logs / f"{benchmark_name}.log")
    logger.info("Running CS3 benchmark '%s' across %d configs.", benchmark_name, len(config_files))

    run_rows: list[dict[str, Any]] = []
    metrics_paths: list[Path] = []
    for raw_config_path in config_files:
        single_config_path = _resolve_config_path(resolved_root, raw_config_path)
        logger.info("Executing single-model config %s", single_config_path)
        config = read_yaml(single_config_path)
        try:
            summary = _dispatch_single_config(resolved_root, single_config_path)
            metrics_path = Path(summary["metrics_csv"])
            metrics_paths.append(metrics_path)
            run_rows.append(
                {
                    "benchmark_name": benchmark_name,
                    "config_path": str(single_config_path),
                    "experiment_name": str(config.get("experiment_name", "")),
                    "case_study": str(config.get("case_study", "")),
                    "model_family": str(config.get("model", {}).get("family", "")),
                    "algorithm_name": str(config.get("model", {}).get("name", "")),
                    "modality_setting": "_".join(config.get("modalities", {}).get("selected", [])),
                    "data_source_type": str(config.get("evaluation", {}).get("data_source_type", "synthetic")),
                    "runtime_type": str(config.get("evaluation", {}).get("runtime_type", "software_only")),
                    "model_status": "fully_runnable",
                    "evidence_level": str(config.get("evaluation", {}).get("evidence_level", "benchmark_preliminary")),
                    "metrics_csv": str(metrics_path),
                    "summary_json": str(summary.get("summary_json", summary.get("config_path", ""))),
                    "status": "completed",
                }
            )
        except Exception as exc:
            logger.warning("Benchmark config %s failed: %s", single_config_path, exc)
            run_rows.append(
                {
                    "benchmark_name": benchmark_name,
                    "config_path": str(single_config_path),
                    "experiment_name": str(config.get("experiment_name", "")),
                    "case_study": str(config.get("case_study", "")),
                    "model_family": str(config.get("model", {}).get("family", "")),
                    "algorithm_name": str(config.get("model", {}).get("name", "")),
                    "modality_setting": "_".join(config.get("modalities", {}).get("selected", [])),
                    "data_source_type": str(config.get("evaluation", {}).get("data_source_type", "synthetic")),
                    "runtime_type": str(config.get("evaluation", {}).get("runtime_type", "software_only")),
                    "model_status": "optional_not_installed",
                    "evidence_level": str(config.get("evaluation", {}).get("evidence_level", "benchmark_preliminary")),
                    "metrics_csv": "",
                    "summary_json": "",
                    "status": f"failed: {exc}",
                }
            )
            if not continue_on_error:
                raise

    master_df = merge_result_tables(metrics_paths)
    if not master_df.empty:
        master_df.insert(0, "rank", range(1, len(master_df) + 1))
    ablation_df = master_df[
        [
            column
            for column in (
                "experiment_name",
                "algorithm_name",
                "model_family",
                "modality_setting",
                "data_source_type",
                "runtime_type",
                "model_status",
                "evidence_level",
                "accuracy",
                "macro_f1",
                "weighted_f1",
                "uar",
                "inference_latency_ms",
            )
            if column in master_df.columns
        ]
    ].copy()

    summary_table_path = paths.outputs_tables / "cs3_master_model_summary.csv"
    ablation_table_path = paths.outputs_tables / "cs3_ablation_summary.csv"
    run_manifest_path = paths.outputs_tables / "cs3_benchmark_run_manifest.csv"
    benchmark_snapshot_path = paths.outputs_tables / f"{benchmark_name}_config_snapshot.yaml"
    benchmark_summary_json_path = paths.outputs_tables / f"{benchmark_name}_summary.json"

    write_dataframe(summary_table_path, master_df)
    write_dataframe(ablation_table_path, ablation_df)
    write_dataframe(run_manifest_path, pd.DataFrame(run_rows))
    write_yaml(benchmark_snapshot_path, benchmark_cfg)

    generate_cs3_figures(
        resolved_root,
        summary_csv=summary_table_path,
        ablation_csv=ablation_table_path,
        output_dir=paths.outputs_figures_cs3,
    )

    summary_payload = {
        "benchmark_name": benchmark_name,
        "config_path": str(config_path.resolve()),
        "master_summary_csv": str(summary_table_path),
        "ablation_summary_csv": str(ablation_table_path),
        "run_manifest_csv": str(run_manifest_path),
        "model_comparison_figure": str(paths.outputs_figures_cs3 / "model_comparison_barplot.png"),
        "ablation_figure": str(paths.outputs_figures_cs3 / "ablation_comparison.png"),
        "latency_figure": str(paths.outputs_figures_cs3 / "inference_latency_comparison.png"),
    }
    write_json(benchmark_summary_json_path, summary_payload)
    logger.info("Finished CS3 benchmark '%s'.", benchmark_name)
    return {key: str(value) for key, value in summary_payload.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CS3 multimodel benchmark over single-model configs.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default="configs/cs3/benchmark_all.yaml")
    args = parser.parse_args()
    run_cs3_benchmark(Path(args.project_root).resolve(), Path(args.config).resolve())


if __name__ == "__main__":
    main()
