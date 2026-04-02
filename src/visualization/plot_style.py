from __future__ import annotations

from pathlib import Path


PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 12.5,
    "axes.titlesize": 16.5,
    "axes.labelsize": 13.5,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 180,
    "savefig.dpi": 300,
}


COLOR_PALETTE = {
    "navy": "#133c55",
    "teal": "#2a9d8f",
    "gold": "#e9c46a",
    "orange": "#f4a261",
    "red": "#e76f51",
    "slate": "#5c677d",
    "mint": "#88c0b7",
    "light": "#f7f7f7",
}


def apply_publication_style(matplotlib_module) -> None:
    matplotlib_module.rcParams.update(PLOT_STYLE)


def save_figure_bundle(fig, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        fig.savefig(output_base.with_suffix(suffix), bbox_inches="tight")
