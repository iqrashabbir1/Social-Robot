from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


BASE_URL = "https://zenodo.org/records/1188976/files/{name}?download=1"


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, dest.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a small RAVDESS video-speech subset for Paper 1.")
    parser.add_argument("--output-root", required=True, help="Target directory, e.g. data/public/RAVDESS")
    parser.add_argument("--actors", nargs="+", default=["01", "02"], help="Actor IDs to download, e.g. 01 02")
    parser.add_argument("--keep-zips", action="store_true", help="Keep downloaded zip files after extraction")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for actor in args.actors:
        actor_id = f"{int(actor):02d}"
        zip_name = f"Video_Speech_Actor_{actor_id}.zip"
        zip_path = output_root / zip_name
        if not zip_path.exists():
            print(f"Downloading {zip_name} ...")
            download_file(BASE_URL.format(name=zip_name), zip_path)
        extract_dir = output_root / f"Actor_{actor_id}"
        if not extract_dir.exists():
            extract_dir.mkdir(parents=True, exist_ok=True)
            print(f"Extracting {zip_name} ...")
            with zipfile.ZipFile(zip_path, "r") as archive:
                archive.extractall(extract_dir)
        if not args.keep_zips and zip_path.exists():
            zip_path.unlink()

    print(output_root)


if __name__ == "__main__":
    main()
