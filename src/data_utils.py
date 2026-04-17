from __future__ import annotations

import json
import random
import re
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import py7zr

ALFRED_LITE_URL = "https://ai2-vision-alfred.s3-us-west-2.amazonaws.com/json_2.1.0.7z"
DEFAULT_SAMPLE_SIZES = {
    "train": 120,
    "valid_seen": 40,
    "valid_unseen": 40,
}


def ensure_alfred_subset(
    data_dir: Path,
    sample_sizes: dict[str, int] | None = None,
    seed: int = 7,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sample_sizes = sample_sizes or DEFAULT_SAMPLE_SIZES
    raw_dir = data_dir / "raw"
    archive_path = raw_dir / "json_2.1.0.7z"
    extracted_dir = raw_dir / "json_2.1.0"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    if not extracted_dir.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        if not archive_path.exists():
            _download_file(ALFRED_LITE_URL, archive_path)
        with py7zr.SevenZipFile(archive_path, mode="r") as archive:
            archive.extractall(path=raw_dir)

    records, summary = build_subset_records(extracted_dir, sample_sizes=sample_sizes, seed=seed)
    output_path = processed_dir / "alfred_subset.json"
    output_path.write_text(json.dumps(records, indent=2))
    return records, summary


def build_subset_records(
    extracted_dir: Path,
    sample_sizes: dict[str, int],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    random_generator = random.Random(seed)
    split_files: dict[str, list[Path]] = {}

    for split, sample_size in sample_sizes.items():
        split_root = extracted_dir / split
        traj_files = sorted(split_root.glob("*/*/traj_data.json"))
        if not traj_files:
            raise FileNotFoundError(f"No ALFRED trajectory files found under {split_root}")
        random_generator.shuffle(traj_files)
        split_files[split] = traj_files[:sample_size]

    records: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"source_dir": str(extracted_dir), "splits": {}}

    for split, files in split_files.items():
        split_records = []
        for path in files:
            trajectory = json.loads(path.read_text())
            split_records.extend(_records_from_trajectory(trajectory, split))
        records.extend(split_records)
        summary["splits"][split] = {
            "trajectory_files": len(files),
            "records": len(split_records),
        }

    summary["total_records"] = len(records)
    summary["instruction_types"] = _count_by_key(records, "instruction_type")
    return records, summary


def _records_from_trajectory(trajectory: dict[str, Any], split: str) -> list[dict[str, Any]]:
    plan = trajectory.get("plan", {}).get("high_pddl", [])
    grouped_steps: dict[int, list[dict[str, str]]] = defaultdict(list)
    full_sequence: list[dict[str, str]] = []

    for entry in plan:
        high_idx = int(entry.get("high_idx", -1))
        discrete_action = entry.get("discrete_action", {})
        step = {
            "action": str(discrete_action.get("action", "")).strip(),
            "object": _serialize_args(discrete_action.get("args", [])),
        }
        if step["action"] and step["action"] != "NoOp":
            full_sequence.append(step)
            grouped_steps[high_idx].append(step)

    annotations = trajectory.get("turk_annotations", {}).get("anns", [])
    task_id = trajectory.get("task_id", "")
    task_type = trajectory.get("task_type", "")
    records: list[dict[str, Any]] = []

    for annotation_index, annotation in enumerate(annotations):
        task_desc = _normalize_instruction(annotation.get("task_desc", ""))
        if task_desc and full_sequence:
            records.append(
                {
                    "record_id": f"{task_id}_ann{annotation_index}_goal",
                    "task_id": task_id,
                    "split": split,
                    "task_type": task_type,
                    "instruction_type": "multi_step",
                    "instruction": task_desc,
                    "target_actions": full_sequence,
                    "step_count": len(full_sequence),
                }
            )

        for high_idx, high_desc in enumerate(annotation.get("high_descs", [])):
            step_instruction = _normalize_instruction(high_desc)
            step_actions = grouped_steps.get(high_idx, [])
            if step_instruction and step_actions:
                records.append(
                    {
                        "record_id": f"{task_id}_ann{annotation_index}_step{high_idx}",
                        "task_id": task_id,
                        "split": split,
                        "task_type": task_type,
                        "instruction_type": "simple" if len(step_actions) == 1 else "multi_step",
                        "instruction": step_instruction,
                        "target_actions": step_actions,
                        "step_count": len(step_actions),
                    }
                )

    return records


def _serialize_args(args: Any) -> str:
    if isinstance(args, list):
        values = [str(value).strip().lower() for value in args if str(value).strip()]
    elif isinstance(args, dict):
        values = [str(value).strip().lower() for value in args.values() if str(value).strip()]
    elif args:
        values = [str(args).strip().lower()]
    else:
        values = []
    return " -> ".join(values)


def _normalize_instruction(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text.strip())
    return cleaned


def _count_by_key(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[str(record[key])] += 1
    return dict(sorted(counts.items()))


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
