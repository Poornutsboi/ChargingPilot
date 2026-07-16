"""Train a hierarchical PPO charging-allocation policy on generated RoadMap data."""

from __future__ import annotations

import argparse
from pathlib import Path

from chargingpilot.cli import run_hierarchical_training
from chargingpilot.paths import roadmap_links_path, roadmap_nodes_path, repository_root


def main() -> None:
    root = repository_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--flows", type=Path, required=True)
    parser.add_argument("--station-config", type=Path, default=root / "configs" / "setting_72stations_roadmap_pv_ess.yaml")
    parser.add_argument("--output-dir", type=Path, default=root / "artifacts")
    parser.add_argument("--total-decisions", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir = args.requests_dir
    args.request_manifest = args.manifest
    args.station_setting = args.station_config
    args.road_map_nodes = roadmap_nodes_path()
    args.road_map_links = roadmap_links_path()
    args.link_flows = args.flows
    args.route_cache = args.output_dir / "route_cache.json"
    args.output = args.output_dir / "roadmap_ppo.pt"
    args.selection_record = args.output_dir / "selected_checkpoint.txt"
    args.request_split = "train"
    args.detour_limit = 0.60
    args.resume = None
    args.use_popart = False
    args.post_selection_test = False
    args.formal_short_run = False
    args.preflight_decisions = min(8, args.rollout_steps)

    result = run_hierarchical_training(args)
    print(result.checkpoint_path)


if __name__ == "__main__":
    main()
