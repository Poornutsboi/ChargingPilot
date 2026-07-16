# Standalone RoadMap PPO Design

## Goal

Make ChargingPilot independently train a charging-allocation policy for the
bundled road-map topology. A fresh clone must not require files, imports, or
runtime paths from `renewable-aware-split-charging`.

## Supported pipeline

```text
data/roadmap/nodes_final.geojson + links_final.geojson
  -> scripts/generate_flows.py
  -> scripts/generate_requests.py
  -> scripts/build_manifest.py
  -> scripts/train_roadmap_ppo.py
  -> artifacts/roadmap_ppo.pt
```

Every generated file is placed beneath a caller-selected `--output-dir`; model
artifacts default below the ignored `artifacts/` directory.

## Runtime boundaries

- `chargingpilot.roadmap`: immutable topology discovery and reproducible flow
  and request generation.
- `chargingpilot.routing`: route distance, feasible charging plans, and cache
  generation using only `chargingpilot.roadmap`.
- `chargingpilot.simulator`: vehicle, station, battery, and interval state;
  remove unused legacy integrations requiring `data.network`.
- `chargingpilot.environment`: map-aware Gymnasium environment with explicit
  configuration and generated input paths.
- `chargingpilot.trainer`: hierarchical PPO policy and trainer used for the
  RoadMap charging-allocation action space. Flat PPO remains an optional
  baseline only when it can use bundled inputs.
- `scripts`: small command-line entry points only; all shared logic stays in
  the package.

## Training inputs

`build_manifest.py` takes generated request CSV files and emits a manifest that
maps train, validation, and test splits. `train_roadmap_ppo.py` accepts the
topology, generated flows, request directory, manifest, station configuration,
and output directory explicitly. It must not contain defaults beginning with
`datasets/`, `exps/`, or `models/`.

## Validation

An end-to-end test will generate a tiny one-day request set and manifest in a
temporary directory, construct the RoadMap environment, collect a very short
hierarchical PPO rollout, and save a checkpoint. The test verifies that every
path resolves within the standalone checkout or the temporary output directory.

## Exclusions

- Existing checkpoints, experiment logs, SwanLab records, and generated
  training datasets.
- Legacy `data.network` integrations.
- Any relative dependency on the original repository.
