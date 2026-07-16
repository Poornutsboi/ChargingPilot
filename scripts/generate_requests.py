"""Generate map-aware charging requests into an explicit output directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from chargingpilot.roadmap.generate_charging_requests import main as generate_requests


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--requests-per-day", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = ROOT / "data" / "roadmap"
    generate_requests(
        [
            "--nodes", str(data_dir / "nodes_final.geojson"),
            "--links", str(data_dir / "links_final.geojson"),
            "--output", str(args.output_dir / "charging_requests.csv"),
            "--days", str(args.days),
            "--requests-per-day", str(args.requests_per_day),
            "--seed", str(args.seed),
        ]
    )


if __name__ == "__main__":
    main()
