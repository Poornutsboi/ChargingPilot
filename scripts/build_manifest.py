"""Create a deterministic split manifest from generated request CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

from chargingpilot.roadmap.manifest import build_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(build_manifest(args.requests_dir, args.output))


if __name__ == "__main__":
    main()
