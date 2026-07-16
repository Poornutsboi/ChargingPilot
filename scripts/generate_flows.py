"""Generate road-map traffic flows into an explicit output directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from chargingpilot.roadmap.generate_link_24h_flow import generate_link_flows, write_link_flows


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    map_dir = ROOT / "data" / "roadmap"
    rows = generate_link_flows(map_dir / "nodes_final.geojson", map_dir / "links_final.geojson")
    print(write_link_flows(rows, args.output_dir / "link_24h_flows.csv"))


if __name__ == "__main__":
    main()
