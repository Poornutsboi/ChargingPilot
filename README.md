# ChargingPilot

Road-map-aware EV charging simulation, routing, and PPO training.

## Development

```powershell
py -3.12 -m pip install -e .
py -3.12 -m pytest -q
```

The repository tracks only source code, tests, configuration, and immutable
road-map topology. Generate requests, traffic flows, checkpoints, and logs into
an external or ignored output directory.

```powershell
python scripts/generate_flows.py --output-dir artifacts/roadmap-data
1..3 | ForEach-Object { python scripts/generate_requests.py --output-dir artifacts/roadmap-data/requests --name ("day_{0:00}.csv" -f $_) --days 1 --requests-per-day 100 --seed $_ }
python scripts/build_manifest.py --requests-dir artifacts/roadmap-data/requests --output artifacts/roadmap-data/manifest.yaml
python scripts/train_roadmap_ppo.py --requests-dir artifacts/roadmap-data/requests --manifest artifacts/roadmap-data/manifest.yaml --flows artifacts/roadmap-data/link_24h_flows.csv --output-dir artifacts/roadmap-ppo
```

The final command writes `artifacts/roadmap-ppo/roadmap_ppo.pt`, a hierarchical
PPO policy that selects feasible charging allocations for the bundled RoadMap
network. No file from the original project checkout is required.
