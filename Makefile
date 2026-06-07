PYTHON ?= python
DOCKER_IMAGE ?= paemdt
CONFIG ?= configs/paemdt_full.yaml

.PHONY: setup data train evaluate figures paper docker-build docker-run reproduce

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) scripts/download_ravdess_subset.py --output-root data/public/RAVDESS --actors 01 02 03 04 05 06 07 08
	$(PYTHON) scripts/prepare_ravdess_labels.py --dataset-root data/public/RAVDESS --output-csv data/public/RAVDESS/labels_broad4_angry.csv --target-label-set broad4_angry
	$(PYTHON) scripts/download_cremad_subset.py --output-root data/public/CREMA-D --actor-ids 1001 1002 1003 1004 1005 1006 1007 1008
	$(PYTHON) scripts/prepare_cremad_labels.py --dataset-root data/public/CREMA-D --output-csv data/public/CREMA-D/labels_broad4_angry.csv --target-label-set broad4_angry

train:
	$(PYTHON) scripts/train_paemdt_full.py --project-root . --config $(CONFIG)

evaluate:
	$(PYTHON) scripts/train_paemdt_full.py --project-root . --config $(CONFIG)

figures:
	$(PYTHON) scripts/generate_paper_tables.py --project-root . --config $(CONFIG)

paper:
	$(PYTHON) scripts/generate_paper_tables.py --project-root . --config $(CONFIG)

docker-build:
	docker build -t $(DOCKER_IMAGE) .

docker-run:
	docker run --rm $(DOCKER_IMAGE)

reproduce:
	$(PYTHON) scripts/train_paemdt_full.py --project-root . --config $(CONFIG) --reproduce
