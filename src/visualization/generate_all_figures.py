from __future__ import annotations

import argparse
from pathlib import Path

from src.visualization.plot_cs1 import generate_cs1_figures
from src.visualization.plot_cs2 import generate_cs2_figures
from src.visualization.plot_cs3 import generate_cs3_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all Paper 1 figures from CSV inputs.")
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    generate_cs1_figures(project_root)
    generate_cs2_figures(project_root)
    generate_cs3_figures(project_root)


if __name__ == "__main__":
    main()
