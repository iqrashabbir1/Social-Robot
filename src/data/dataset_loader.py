from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.public_dataset_presets import map_to_target_label


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv"}


@dataclass(frozen=True)
class DatasetRecord:
    sample_id: str
    media_path: str
    media_type: str
    label: str | None
    split: str
    timestamp_ms: float | None
    frame_index: int | None
    source_type: str


def normalize_dataset_label(raw_label: str | None) -> str | None:
    return map_to_target_label(raw_label, None)


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS.union(VIDEO_EXTENSIONS):
            yield path


def _from_folder_structure(dataset_root: Path) -> list[DatasetRecord]:
    records: list[DatasetRecord] = []
    for path in _iter_files(dataset_root):
        label = None
        if path.parent != dataset_root:
            label = normalize_dataset_label(path.parent.name)
        media_type = "image" if path.suffix.lower() in IMAGE_EXTENSIONS else "video"
        records.append(
            DatasetRecord(
                sample_id=path.stem,
                media_path=str(path.resolve()),
                media_type=media_type,
                label=label,
                split="all",
                timestamp_ms=None,
                frame_index=None,
                source_type="folder_structure",
            )
        )
    return records


def _from_labels_csv(dataset_root: Path, labels_csv: Path) -> list[DatasetRecord]:
    labels_df = pd.read_csv(labels_csv)
    required_cols = {"path"}
    if not required_cols.issubset(labels_df.columns):
        raise ValueError(f"labels_csv must contain at least these columns: {sorted(required_cols)}")
    records: list[DatasetRecord] = []
    for row in labels_df.to_dict(orient="records"):
        media_path = Path(str(row["path"]))
        if not media_path.is_absolute():
            media_path = dataset_root / media_path
        media_type = "image" if media_path.suffix.lower() in IMAGE_EXTENSIONS else "video"
        records.append(
            DatasetRecord(
                sample_id=str(row.get("sample_id", media_path.stem)),
                media_path=str(media_path.resolve()),
                media_type=media_type,
                label=normalize_dataset_label(row.get("label")),
                split=str(row.get("split", "all")),
                timestamp_ms=float(row["timestamp_ms"]) if row.get("timestamp_ms") not in (None, "") else None,
                frame_index=int(row["frame_index"]) if row.get("frame_index") not in (None, "") else None,
                source_type="labels_csv",
            )
        )
    return records


def load_dataset_records(
    dataset_root: Path,
    labels_csv: Path | None = None,
    split_mode: str = "test_only",
    test_size: float = 0.2,
    random_seed: int = 42,
    target_label_set: str | None = None,
) -> pd.DataFrame:
    dataset_root = dataset_root.resolve()
    if labels_csv is not None and labels_csv.exists():
        records = _from_labels_csv(dataset_root, labels_csv.resolve())
    else:
        records = _from_folder_structure(dataset_root)
    if not records:
        raise FileNotFoundError(f"No dataset images or videos were found under {dataset_root}")
    df = pd.DataFrame(asdict(record) for record in records)
    if target_label_set:
        df["label"] = df["label"].map(lambda value: map_to_target_label(value, target_label_set))
    if split_mode == "test_only":
        df["split"] = "test"
        return df
    if split_mode == "train_test":
        labeled_mask = df["label"].notna()
        labeled = df.loc[labeled_mask].copy()
        unlabeled = df.loc[~labeled_mask].copy()
        if labeled.empty:
            df["split"] = "test"
            return df
        stratify = labeled["label"] if labeled["label"].nunique() > 1 else None
        train_idx, test_idx = train_test_split(
            labeled.index.tolist(),
            test_size=test_size,
            random_state=random_seed,
            stratify=stratify,
        )
        df["split"] = "unused"
        df.loc[train_idx, "split"] = "train"
        df.loc[test_idx, "split"] = "test"
        if not unlabeled.empty:
            df.loc[unlabeled.index, "split"] = "test"
        return df
    raise ValueError("split_mode must be one of: test_only, train_test")


def load_image_frame(path: Path, width: int | None = None, height: int | None = None):
    frame = cv2.imread(str(path))
    if frame is None:
        raise ValueError(f"Could not read image: {path}")
    if width and height:
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame


def materialize_frame_records(
    dataset_df: pd.DataFrame,
    cache_dir: Path,
    width: int | None = None,
    height: int | None = None,
    video_frame_stride: int = 15,
    max_frames_per_video: int = 12,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    materialized_rows: list[dict[str, object]] = []
    for row in dataset_df.to_dict(orient="records"):
        media_path = Path(row["media_path"])
        if row["media_type"] == "image":
            materialized_rows.append(
                {
                    **row,
                    "frame_path": str(media_path),
                }
            )
            continue

        capture = cv2.VideoCapture(str(media_path))
        if not capture.isOpened():
            continue
        frame_counter = 0
        saved_counter = 0
        while saved_counter < max_frames_per_video:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            if frame_counter % max(1, video_frame_stride) == 0:
                if width and height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                frame_path = cache_dir / f"{media_path.stem}_frame_{saved_counter:04d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                materialized_rows.append(
                    {
                        **row,
                        "sample_id": f"{row['sample_id']}_frame_{saved_counter:04d}",
                        "frame_path": str(frame_path.resolve()),
                        "frame_index": saved_counter,
                        "timestamp_ms": float((capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)),
                        "media_type": "image",
                    }
                )
                saved_counter += 1
            frame_counter += 1
        capture.release()
    return pd.DataFrame(materialized_rows)
