from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CacheRecord:
    origin: int
    destination: int
    station_id: int
    distance_m: float


def build_cache_key(
    road_map: Any,
    node_ids: tuple[int, ...],
    *,
    directed: bool,
    station_ids: tuple[int, ...],
) -> str:
    edges = []
    for source in node_ids:
        for target, weight in road_map.graph.get(source, ()):
            edge = (int(source), int(target))
            length = road_map.edge_lengths_m.get(edge)
            edges.append(
                [
                    int(source),
                    int(target),
                    float(weight),
                    None if length is None else float(length),
                ]
            )
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "node_ids": list(node_ids),
        "edges": sorted(edges),
        "directed": bool(directed),
        "station_ids": list(station_ids),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_cache(path: Path) -> tuple[str, tuple[CacheRecord, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported or missing cache schema version")
    key = payload.get("key")
    raw_entries = payload.get("entries")
    if not isinstance(key, str) or not isinstance(raw_entries, list):
        raise ValueError("cache key or entries are missing")
    records = []
    od_pairs: set[tuple[int, int]] = set()
    for raw in raw_entries:
        if not isinstance(raw, dict):
            raise ValueError("cache entry must be an object")
        try:
            record = CacheRecord(
                origin=int(raw["origin"]),
                destination=int(raw["destination"]),
                station_id=int(raw["station_id"]),
                distance_m=float(raw["distance_m"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid cache entry") from exc
        if record.distance_m < 0 or not _is_finite(record.distance_m):
            raise ValueError("cache distance must be finite and non-negative")
        od = (record.origin, record.destination)
        if od in od_pairs:
            raise ValueError(f"duplicate cache entry for {od}")
        od_pairs.add(od)
        records.append(record)
    return key, tuple(records)


def write_cache(path: Path, key: str, records: tuple[CacheRecord, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "key": key,
        "entries": [
            asdict(record)
            for record in sorted(
                records,
                key=lambda item: (item.origin, item.destination),
            )
        ],
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _is_finite(value: float) -> bool:
    return value != float("inf") and value != float("-inf") and value == value
