from pathlib import Path

import yaml


def test_topology_paths_resolve_inside_checkout() -> None:
    from chargingpilot.paths import roadmap_data_dir

    data_dir = roadmap_data_dir()

    assert data_dir.name == "roadmap"
    assert (data_dir / "nodes_final.geojson").is_file()
    assert (data_dir / "links_final.geojson").is_file()


def test_manifest_builder_creates_non_overlapping_splits(tmp_path: Path) -> None:
    from chargingpilot.roadmap.manifest import build_manifest

    requests_dir = tmp_path / "requests"
    requests_dir.mkdir()
    for index in range(5):
        (requests_dir / f"day_{index:02d}.csv").write_text("vehicle_id\nEV1\n", encoding="utf-8")

    manifest_path = build_manifest(requests_dir, tmp_path / "manifest.yaml")
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    assert set(manifest) == {"train", "validation", "test"}
    assert all(manifest[split] for split in manifest)
    assert len(set().union(*map(set, manifest.values()))) == 5
