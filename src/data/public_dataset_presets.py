from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.io_utils import write_dataframe


TARGET_LABEL_SETS: dict[str, set[str]] = {
    "broad4_angry": {"happy", "sad", "neutral", "angry"},
    "paper1_legacy4": {"happy", "sad", "neutral", "fear"},
}


def canonicalize_emotion_label(raw_label: str | None) -> str | None:
    if raw_label is None:
        return None
    value = str(raw_label).strip().lower()
    alias_map = {
        "happy": "happy",
        "happiness": "happy",
        "joy": "happy",
        "hap": "happy",
        "sad": "sad",
        "sadness": "sad",
        "angry": "angry",
        "anger": "angry",
        "ang": "angry",
        "neutral": "neutral",
        "neu": "neutral",
        "fear": "fear",
        "fearful": "fear",
        "fea": "fear",
        "surprise": "surprise",
        "surprised": "surprise",
        "sur": "surprise",
        "disgust": "disgust",
        "dis": "disgust",
        "contempt": "contempt",
        "con": "contempt",
        "none": None,
        "unknown": None,
        "unlabeled": None,
    }
    return alias_map.get(value, value)


def map_to_target_label(raw_label: str | None, target_label_set: str | None = None) -> str | None:
    canonical = canonicalize_emotion_label(raw_label)
    if canonical is None:
        return None
    if not target_label_set:
        return canonical
    allowed = TARGET_LABEL_SETS.get(target_label_set)
    if allowed is None:
        raise ValueError(f"Unknown target_label_set: {target_label_set}")
    return canonical if canonical in allowed else None


def _resolve_rafdb_image(dataset_root: Path, image_name: str) -> Path | None:
    stem = Path(image_name).stem
    candidates = [
        dataset_root / "Image" / "aligned" / f"{stem}_aligned.jpg",
        dataset_root / "Image" / "aligned" / f"{stem}_aligned.png",
        dataset_root / "Image" / "original" / image_name,
        dataset_root / image_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def build_rafdb_labels_csv(
    dataset_root: Path,
    output_csv: Path,
    target_label_set: str = "broad4_angry",
    annotation_file: Path | None = None,
) -> Path:
    dataset_root = dataset_root.resolve()
    annotation_candidates = [
        dataset_root / "EmoLabel" / "list_patition_label.txt",
        dataset_root / "basic" / "EmoLabel" / "list_patition_label.txt",
    ]
    annotation_path = annotation_file.resolve() if annotation_file else next((p for p in annotation_candidates if p.exists()), None)
    if annotation_path is None:
        raise FileNotFoundError("RAF-DB annotation file was not found. Expected list_patition_label.txt under EmoLabel/")

    raf_label_map = {
        "1": "surprise",
        "2": "fear",
        "3": "disgust",
        "4": "happy",
        "5": "sad",
        "6": "angry",
        "7": "neutral",
    }

    rows: list[dict[str, object]] = []
    for line in annotation_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        image_name, raw_code = parts[0], parts[1]
        resolved_path = _resolve_rafdb_image(dataset_root, image_name)
        if resolved_path is None:
            continue
        raw_label = raf_label_map.get(raw_code)
        mapped_label = map_to_target_label(raw_label, target_label_set)
        if mapped_label is None:
            continue
        split = "test" if image_name.lower().startswith("test") else "train" if image_name.lower().startswith("train") else "all"
        rows.append(
            {
                "sample_id": Path(image_name).stem,
                "path": str(resolved_path),
                "label": mapped_label,
                "split": split,
                "source_dataset": "RAF-DB",
                "target_label_set": target_label_set,
                "raw_label": raw_label,
            }
        )
    if not rows:
        raise RuntimeError("No RAF-DB samples were mapped into the requested target label set.")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_dataframe(output_csv, pd.DataFrame(rows))
    return output_csv


def build_cremad_labels_csv(
    dataset_root: Path,
    output_csv: Path,
    target_label_set: str = "broad4_angry",
    include_video: bool = True,
    include_audio: bool = False,
) -> Path:
    dataset_root = dataset_root.resolve()
    emotion_code_map = {
        "ANG": "angry",
        "DIS": "disgust",
        "FEA": "fear",
        "HAP": "happy",
        "NEU": "neutral",
        "SAD": "sad",
    }
    video_exts = {".avi", ".mp4", ".mov", ".flv", ".mkv"}
    audio_exts = {".wav"}

    rows: list[dict[str, object]] = []
    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in video_exts and not include_video:
            continue
        if suffix in audio_exts and not include_audio:
            continue
        if suffix not in video_exts.union(audio_exts):
            continue
        parts = path.stem.split("_")
        if len(parts) < 3:
            continue
        raw_label = emotion_code_map.get(parts[2].upper())
        mapped_label = map_to_target_label(raw_label, target_label_set)
        if mapped_label is None:
            continue
        rows.append(
            {
                "sample_id": path.stem,
                "path": str(path.resolve()),
                "label": mapped_label,
                "split": "test",
                "source_dataset": "CREMA-D",
                "target_label_set": target_label_set,
                "raw_label": raw_label,
            }
        )
    if not rows:
        raise RuntimeError("No CREMA-D samples were mapped into the requested target label set.")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_dataframe(output_csv, pd.DataFrame(rows))
    return output_csv


def build_emorynlp_labels_csv(
    dataset_root: Path,
    annotation_csv: Path,
    output_csv: Path,
    target_label_set: str = "broad4_angry",
) -> Path:
    dataset_root = dataset_root.resolve()
    annotation_csv = annotation_csv.resolve()
    annotations_df = pd.read_csv(annotation_csv)

    label_map = {
        "joyful": "happy",
        "mad": "angry",
        "neutral": "neutral",
        "sad": "sad",
        "sadness": "sad",
    }
    split_name = annotation_csv.stem.lower()
    split = "test" if "test" in split_name else "train" if "train" in split_name else "dev"

    rows: list[dict[str, object]] = []
    for row in annotations_df.to_dict(orient="records"):
        raw_label = label_map.get(str(row.get("Emotion", "")).strip().lower())
        mapped_label = map_to_target_label(raw_label, target_label_set)
        if mapped_label is None:
            continue
        season = row.get("Season")
        episode = row.get("Episode")
        scene_id = row.get("Scene_ID")
        utterance_id = row.get("Utterance_ID")
        if any(value in (None, "") for value in (season, episode, scene_id, utterance_id)):
            continue
        filename = f"sea{int(season)}_ep{int(episode)}_sc{int(scene_id)}_utt{int(utterance_id)}.mp4"
        candidates = [dataset_root / filename]
        candidates.extend(dataset_root.rglob(filename))
        resolved_path = next((candidate.resolve() for candidate in candidates if candidate.exists()), None)
        if resolved_path is None:
            continue
        rows.append(
            {
                "sample_id": Path(filename).stem,
                "path": str(resolved_path),
                "label": mapped_label,
                "split": split,
                "source_dataset": "EmoryNLP",
                "target_label_set": target_label_set,
                "raw_label": raw_label,
            }
        )
    if not rows:
        raise RuntimeError("No EmoryNLP samples were mapped into the requested target label set.")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_dataframe(output_csv, pd.DataFrame(rows))
    return output_csv


def build_ravdess_labels_csv(
    dataset_root: Path,
    output_csv: Path,
    target_label_set: str = "broad4_angry",
    full_av_only: bool = True,
    speech_only: bool = True,
) -> Path:
    dataset_root = dataset_root.resolve()
    ravdess_label_map = {
        "01": "neutral",
        "02": "neutral",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fear",
        "07": "disgust",
        "08": "surprise",
    }

    rows: list[dict[str, object]] = []
    for path in sorted(dataset_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".mp4", ".avi", ".wav"}:
            continue
        parts = path.stem.split("-")
        if len(parts) != 7:
            continue
        modality, channel, emotion_code = parts[0], parts[1], parts[2]
        if full_av_only and modality != "01":
            continue
        if speech_only and channel != "01":
            continue
        raw_label = ravdess_label_map.get(emotion_code)
        mapped_label = map_to_target_label(raw_label, target_label_set)
        if mapped_label is None:
            continue
        rows.append(
            {
                "sample_id": path.stem,
                "path": str(path.resolve()),
                "label": mapped_label,
                "split": "test",
                "source_dataset": "RAVDESS",
                "target_label_set": target_label_set,
                "raw_label": raw_label,
            }
        )
    if not rows:
        raise RuntimeError("No RAVDESS samples were mapped into the requested target label set.")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_dataframe(output_csv, pd.DataFrame(rows))
    return output_csv
