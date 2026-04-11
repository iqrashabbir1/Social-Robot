from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


REPO_URL = "https://github.com/CheyneyComputerScience/CREMA-D.git"


def _run(command: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def download_cremad_subset(output_root: Path, actor_ids: list[str], keep_repo_cache: bool = True) -> Path:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    cache_repo = output_root / "_cremad_repo_cache"
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"

    if not cache_repo.exists():
        _run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", REPO_URL, str(cache_repo)],
            env=env,
        )

    _run(["git", "lfs", "install", "--local"], cwd=cache_repo)
    _run(["git", "sparse-checkout", "init", "--no-cone"], cwd=cache_repo)

    normalized_ids = [str(actor_id).strip() for actor_id in actor_ids if str(actor_id).strip()]
    sparse_lines = [".gitattributes", "processedResults"]
    sparse_lines.extend(f"VideoFlash/{actor_id}_*" for actor_id in normalized_ids)
    sparse_lines.extend(f"AudioWAV/{actor_id}_*" for actor_id in normalized_ids)
    sparse_path = cache_repo / ".git" / "info" / "sparse-checkout"
    sparse_path.write_text("\n".join(sparse_lines) + "\n", encoding="utf-8")

    _run(["git", "read-tree", "-mu", "HEAD"], cwd=cache_repo)

    include_patterns = [f"VideoFlash/{actor_id}_*" for actor_id in normalized_ids]
    include_patterns.extend(f"AudioWAV/{actor_id}_*" for actor_id in normalized_ids)
    _run(
        ["git", "lfs", "pull", "--include", ",".join(include_patterns)],
        cwd=cache_repo,
    )

    for directory_name in ("VideoFlash", "AudioWAV", "processedResults"):
        source_dir = cache_repo / directory_name
        if not source_dir.exists():
            continue
        target_dir = output_root / directory_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for item in source_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, target_dir / item.name)

    if not keep_repo_cache and cache_repo.exists():
        shutil.rmtree(cache_repo, ignore_errors=True)

    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a sparse CREMA-D subset for Paper 1 testing.")
    parser.add_argument("--output-root", required=True, help="Target directory, e.g. data/public/CREMA-D")
    parser.add_argument(
        "--actor-ids",
        nargs="+",
        default=["1001", "1002", "1003", "1004", "1005", "1006"],
        help="CREMA-D actor IDs to include, e.g. 1001 1002 1003",
    )
    parser.add_argument("--drop-cache", action="store_true", help="Remove the intermediate git cache after download")
    args = parser.parse_args()

    output_root = download_cremad_subset(
        output_root=Path(args.output_root),
        actor_ids=args.actor_ids,
        keep_repo_cache=not args.drop_cache,
    )
    print(output_root)


if __name__ == "__main__":
    main()

