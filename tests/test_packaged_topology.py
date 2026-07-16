import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_packaged_topology_contains_toll_and_service_nodes() -> None:
    nodes_path = ROOT / "data" / "roadmap" / "nodes_final.geojson"

    nodes = json.loads(nodes_path.read_text(encoding="utf-8"))
    types = {feature["properties"]["type"] for feature in nodes["features"]}

    assert {"toll", "service"} <= types
