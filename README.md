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
python scripts/train_roadmap_ppo.py --output-dir artifacts/roadmap-ppo
```

The command reads the fixed request split and traffic-flow data tracked under
`data/`, then writes `artifacts/roadmap-ppo/roadmap_ppo.pt`, a hierarchical PPO
policy that selects feasible charging allocations for the bundled RoadMap
network. No file from the original project checkout is required.
