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
python scripts/generate_flows.py --output-dir .tmp-flows
python scripts/generate_requests.py --output-dir .tmp-requests --days 1 --requests-per-day 10
```
