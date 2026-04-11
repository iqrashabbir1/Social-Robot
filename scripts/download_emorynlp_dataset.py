from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


DRIVE_URL = "https://drive.google.com/uc?id=1UQduKw8QTqGf3RafxrTDfI1NyInYK3fr"
ANNOTATION_URLS = {
    "train": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/emorynlp/train_sent_emo.csv",
    "dev": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/emorynlp/dev_sent_emo.csv",
    "test": "https://raw.githubusercontent.com/declare-lab/MELD/master/data/emorynlp/test_sent_emo.csv",
}


def _download_annotations(output_root: Path) -> None:
    annotations_dir = output_root / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    for split_name, url in ANNOTATION_URLS.items():
        destination = annotations_dir / f"{split_name}_sent_emo.csv"
        if destination.exists():
            continue
        with urllib.request.urlopen(url, timeout=60) as response, destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the official multimodal EmoryNLP dataset and annotations.")
    parser.add_argument("--output-root", required=True, help="Target directory, e.g. data/public/EmoryNLP")
    parser.add_argument("--keep-zip", action="store_true", help="Keep the downloaded zip archive after extraction")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _download_annotations(output_root)

    try:
        import gdown
    except ImportError as exc:  # pragma: no cover - dependency is optional at import time
        raise SystemExit(
            "gdown is required for the EmoryNLP video download. Install it with "
            "'python -m pip install gdown' or from requirements.txt."
        ) from exc

    zip_path = output_root / "emorynlp_videos.zip"
    if not zip_path.exists():
        gdown.download(DRIVE_URL, str(zip_path), quiet=False, fuzzy=True)

    extract_dir = output_root / "videos"
    if not extract_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(extract_dir)

    if not args.keep_zip and zip_path.exists():
        zip_path.unlink()

    print(output_root)


if __name__ == "__main__":
    main()
