# Social Robot

PAEMDT: Privacy-Aware Explainable Multimodal Digital-Twin Cognitive Caregiving Robot.

This repository preserves the stable baseline code paths while adding a publication-oriented experimental layer for:
- multimodal emotion recognition
- domain adaptation
- differential privacy
- repeated cross-validation
- missing-modality robustness
- digital twin predictive validation
- edge deployment benchmarking
- paper table and figure generation

## Installation

### Local Python environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### ROS 2 / Digital-twin stack
The reproducibility container installs ROS 2 Humble on Ubuntu 22.04 using the official deb-package route documented by ROS 2: [ROS 2 Humble Ubuntu installation](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html).

For local non-container ROS work, use the same distribution:
- Ubuntu 22.04
- ROS 2 Humble

## Data Download

The paper workflows use public RAVDESS and CREMA-D subsets already supported by the repository scripts.

### Download RAVDESS subset
```bash
python scripts/download_ravdess_subset.py --output-root data/public/RAVDESS --actors 01 02 03 04 05 06 07 08
python scripts/prepare_ravdess_labels.py --dataset-root data/public/RAVDESS --output-csv data/public/RAVDESS/labels_broad4_angry.csv --target-label-set broad4_angry
```

### Download CREMA-D subset
```bash
python scripts/download_cremad_subset.py --output-root data/public/CREMA-D --actor-ids 1001 1002 1003 1004 1005 1006 1007 1008
python scripts/prepare_cremad_labels.py --dataset-root data/public/CREMA-D --output-csv data/public/CREMA-D/labels_broad4_angry.csv --target-label-set broad4_angry
```

## Training From Scratch

The integrated end-to-end paper pipeline is:
```bash
python scripts/train_paemdt_full.py --project-root . --config configs/paemdt_full.yaml --reproduce
```

This command:
- ensures the public datasets exist
- runs the integrated PAEMDT training/evaluation pipeline
- regenerates paper tables and figures

Main outputs:
- `experiments/results/paemdt_full/`
- `experiments/results/paper_tables/`
- `experiments/figures/paper_tables/`

## Reproducing Paper Results

## Paper 1 Reproducibility and Evidence Alignment

The PAEMDT Paper 1 artifact layer is traceable through explicit evaluation and visualization scripts. The artifact map is available at `docs/paper1/PAPER1_ARTIFACT_MAP.md`, and the evidence boundary is documented in `docs/paper1/LIMITATIONS_AND_EVIDENCE_BOUNDARY.md`.

Core paper-alignment scripts:
- `src/evaluation/run_domain_adaptation.py`
- `src/evaluation/run_dp_privacy_accounting.py`
- `src/evaluation/run_repeated_cv_statistics.py`
- `src/evaluation/run_calibration_analysis.py`
- `src/evaluation/run_missing_modality_robustness.py`
- `src/evaluation/run_privacy_latency_analysis.py`
- `src/evaluation/run_digital_twin_sync_analysis.py`
- `src/evaluation/run_evidence_maturity.py`
- `src/visualization/generate_all_figures.py`

The repository distinguishes implemented real baselines, benchmark-supported experimental modules, simulation-supported modules, prototype modules, and planned clinical validation. Some paper-aligned values are manuscript-facing experimental summaries when full repeated retraining logs or hardware-specific runs are not present locally; these rows are marked with evidence notes in the generated CSV files.

The repository does not yet provide clinical deployment, ethics approval, live wearable integration, bedside hardware integration, assisted-living pilot execution, or clinician-validated prospective outcomes. Field deployment remains future work.

### Generate paper tables and publication figures
```bash
python scripts/generate_paper_tables.py --project-root . --config configs/paemdt_full.yaml
```

### Full reproducibility check
```bash
make reproduce
```

### Docker
Build and run the full package:
```bash
docker build -t paemdt .
docker run --rm paemdt
```

GPU-enabled compose service:
```bash
docker compose -f docker/docker-compose.yml up paemdt-gpu
```

CPU-only edge-simulation service:
```bash
docker compose -f docker/docker-compose.yml up paemdt-cpu
```

## Edge Deployment Guide

Measured and aggregated edge artifacts are generated through:
- `scripts/benchmark_edge.py`
- `scripts/profile_model.py`
- `scripts/generate_benchmark_table.py`

Example benchmark command:
```bash
python scripts/benchmark_edge.py --project-root . --platform-id raspberry_pi_4 --device cpu
```

After device-specific runs:
```bash
python scripts/generate_benchmark_table.py --project-root .
```

## Make Targets

Common entry points:
- `make setup`
- `make data`
- `make train`
- `make evaluate`
- `make figures`
- `make paper`
- `make docker-build`
- `make docker-run`
- `make reproduce`

## Repository Structure

```text
configs/        reproducible experiment and paper configs
data/           public dataset subsets and prepared labels
docker/         container and compose setup
docs/           manuscript and technical documentation
experiments/    reproducible paper results and figures
launch/         ROS 2 launch files
scripts/        orchestration, download, evaluation, and paper export
social_robot/   ROS-facing runtime nodes
src/            training, evaluation, privacy, digital twin, and plotting code
```

## Citation

If you use this repository or its experimental package, cite the manuscript and repository together.

Suggested BibTeX:
```bibtex
@misc{paemdt_social_robot_repo,
  title        = {PAEMDT: Privacy-Aware Explainable Multimodal Digital-Twin Cognitive Caregiving Robot},
  author       = {Shabbir, Iqra and collaborators},
  year         = {2026},
  note         = {Research code repository and reproducibility package},
  howpublished = {\url{https://github.com/iqrashabbir1/Social-Robot}}
}
```

## Notes on Reproducibility

- The integrated pipeline reproduces the software-side paper artifacts from a single entry point.
- Hardware-specific edge results still depend on running the benchmark on the named target platform.
- Public dataset availability depends on the upstream sources remaining accessible.
- The repository separates measured outputs from paper-reference targets where a manuscript table intentionally reports a finalized benchmark target.
