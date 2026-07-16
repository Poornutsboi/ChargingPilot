from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml


_SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class RequestManifest:
    train: tuple[Path, ...]
    validation: tuple[Path, ...]
    test: tuple[Path, ...]

    @classmethod
    def load(cls, path: str | Path, root: str | Path) -> "RequestManifest":
        manifest_path = Path(path)
        request_root = Path(root)
        with manifest_path.open(encoding="utf-8") as yaml_file:
            raw = yaml.safe_load(yaml_file)
        if not isinstance(raw, Mapping):
            raise TypeError("request manifest must be a YAML mapping")
        if set(raw) != set(_SPLIT_NAMES):
            raise ValueError("request manifest must contain exactly train, validation, and test")

        resolved_root = request_root.resolve()
        resolved: dict[str, tuple[Path, ...]] = {}
        seen: set[Path] = set()
        for split_name in _SPLIT_NAMES:
            values = raw[split_name]
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise TypeError(f"request manifest split {split_name!r} must be a list")
            paths: list[Path] = []
            for value in values:
                if not isinstance(value, str):
                    raise TypeError(f"request manifest entries must be strings, got {type(value).__name__}")
                entry_path = Path(value)
                if entry_path.is_absolute():
                    raise ValueError(f"request manifest entry must stay under request root: {value}")
                request_path = request_root / entry_path
                duplicate_key = request_path.resolve(strict=False)
                try:
                    duplicate_key.relative_to(resolved_root)
                except ValueError as exc:
                    raise ValueError(f"request manifest entry must stay under request root: {value}") from exc
                if duplicate_key in seen:
                    raise ValueError(f"duplicate request file in manifest: {value}")
                if not request_path.is_file():
                    raise FileNotFoundError(f"request file does not exist: {request_path}")
                seen.add(duplicate_key)
                paths.append(request_path)
            resolved[split_name] = tuple(paths)
        return cls(**resolved)


__all__ = ["RequestManifest"]
