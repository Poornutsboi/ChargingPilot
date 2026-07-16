"""Deterministic manifests for generated RoadMap request files."""

from __future__ import annotations

from pathlib import Path

import yaml


def build_manifest(requests_dir: Path, output_path: Path) -> Path:
    files = sorted(path for path in requests_dir.glob("*.csv") if path.is_file())
    if len(files) < 3:
        raise ValueError("at least three request CSV files are required for train, validation, and test splits")

    splits = {"train": [], "validation": [], "test": []}
    for index, path in enumerate(files):
        split = ("validation", "test", "train")[index % 5] if index < 2 else "train"
        splits[split].append(path.relative_to(requests_dir).as_posix())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(splits, sort_keys=False), encoding="utf-8")
    return output_path
