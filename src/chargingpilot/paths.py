"""Paths to versioned ChargingPilot assets."""

from __future__ import annotations

from pathlib import Path


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def roadmap_data_dir() -> Path:
    return repository_root() / "data" / "roadmap"


def roadmap_nodes_path() -> Path:
    return roadmap_data_dir() / "nodes_final.geojson"


def roadmap_links_path() -> Path:
    return roadmap_data_dir() / "links_final.geojson"
