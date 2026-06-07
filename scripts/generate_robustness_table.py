from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.io_utils import write_dataframe, write_json
from src.common.paths import Paper1Paths


ORDER = [
    "Full input",
    "Visual dropout",
    "Speech removal",
    "Physio removal",
    "Crowded room (SNR=10 dB)",
    "Crowded room (SNR=5 dB)",
    "Crowded room (SNR=0 dB)",
    "Night monitoring (low-light)",
    "Multi-sensor dropout (2/5)",
    "Multi-sensor dropout (3/5)",
    "All sensors noisy",
]


def _sort_rows(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()
    ranked["sort_key"] = ranked["condition"].map({name: idx for idx, name in enumerate(ORDER)}).fillna(999).astype(int)
    ranked = ranked.sort_values("sort_key").drop(columns=["sort_key"]).reset_index(drop=True)
    return ranked


def _plot_figure8(df: pd.DataFrame, output_base: Path) -> None:
    sns.set_theme(style="whitegrid")
    plot_df = df.copy()
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.6))

    sns.barplot(data=plot_df, x="Condition", y="Macro-F1", ax=axes[0], color="#4477AA")
    axes[0].axhline(0.85, color="#228833", linestyle="--", linewidth=1.2, label="Autonomous threshold")
    axes[0].axhline(0.70, color="#CCBB44", linestyle=":", linewidth=1.2, label="Caregiver review threshold")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Figure 8A. Macro-F1 Under Missing or Degraded Modalities")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Macro-F1")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].legend(loc="lower left", fontsize=8)

    sns.barplot(data=plot_df, x="Condition", y="Escalation (%)", ax=axes[1], color="#CC6677")
    axes[1].set_ylim(0.0, max(60.0, float(plot_df["Escalation (%)"].max()) + 5.0))
    axes[1].set_title("Figure 8B. HITL Escalation Under Modality Degradation")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Escalation rate (%)")
    axes[1].tick_params(axis="x", rotation=35)

    fig.tight_layout()
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def generate_robustness_table(project_root: Path, input_csv: Path | None = None, output_subdir: str = "missing_modality_robustness") -> dict[str, str]:
    paths = Paper1Paths.from_project_root(project_root)
    paths.ensure()
    source_csv = input_csv or (paths.outputs_csv_paper1 / output_subdir / "missing_modality_scenario_metrics.csv")
    summary_df = pd.read_csv(source_csv)
    summary_df = _sort_rows(summary_df)

    pretty_df = pd.DataFrame(
        {
            "Condition": summary_df["condition"],
            "Macro-F1": summary_df["macro_f1"].round(3),
            "Delta from Full": summary_df["delta_from_full"].apply(lambda value: "--" if abs(float(value)) < 1e-9 else f"{float(value):+.3f}"),
            "Escalation (%)": summary_df["hitl_escalation_percent"].round(1),
            "Safety Status": summary_df["safety_status"]
            .astype(str)
            .str.replace("✅ ", "", regex=False)
            .str.replace("⚠️ ", "", regex=False)
            .str.replace("🔴 ", "", regex=False),
            "HITL Policy": summary_df["hitl_policy"],
            "Mask Suppression": summary_df["mask_suppression_success_rate"].round(3),
        }
    )

    table_path = paths.outputs_tables / "paper1_table_missing_modality_robustness.csv"
    figure_csv_path = paths.outputs_csv_paper1 / output_subdir / "figure8_robustness_breakdown.csv"
    write_dataframe(table_path, pretty_df)
    write_dataframe(figure_csv_path, pretty_df)

    figure_base = paths.outputs_figures_paper1 / "robustness_missing_modalities_detailed"
    _plot_figure8(pretty_df, figure_base)

    write_json(
        paths.outputs_csv_paper1 / output_subdir / "figure8_robustness_manifest.json",
        {
            "input_csv": str(source_csv.resolve()),
            "table_csv": str(table_path.resolve()),
            "figure_csv": str(figure_csv_path.resolve()),
            "figure_png": str(figure_base.with_suffix(".png").resolve()),
            "figure_svg": str(figure_base.with_suffix(".svg").resolve()),
        },
    )

    return {
        "table_csv": str(table_path.resolve()),
        "figure_csv": str(figure_csv_path.resolve()),
        "figure_png": str(figure_base.with_suffix(".png").resolve()),
        "figure_svg": str(figure_base.with_suffix(".svg").resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the detailed missing-modality robustness table and Figure 8 assets.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--input-csv", default="")
    parser.add_argument("--output-subdir", default="missing_modality_robustness")
    args = parser.parse_args()

    outputs = generate_robustness_table(
        project_root=Path(args.project_root).resolve(),
        input_csv=Path(args.input_csv).resolve() if args.input_csv else None,
        output_subdir=args.output_subdir,
    )
    print(f"Robustness table: {outputs['table_csv']}")
    print(f"Figure 8 CSV: {outputs['figure_csv']}")
    print(f"Figure 8 PNG: {outputs['figure_png']}")


if __name__ == "__main__":
    main()
