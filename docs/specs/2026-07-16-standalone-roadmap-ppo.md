# Standalone RoadMap PPO Design

## Goal

Make ChargingPilot independently train a charging-allocation policy for the
bundled road-map topology. A fresh clone must not require files, imports, or
runtime paths from `renewable-aware-split-charging`.

## Supported pipeline

```text
data/roadmap/{nodes_final.geojson, links_final.geojson, link_24h_flows.csv}
data/requests/{charge_request_day_*.csv, manifest.yaml}
  -> scripts/train_roadmap_ppo.py
  -> artifacts/roadmap_ppo.pt
```

Request and flow data are fixed, versioned training inputs copied from the
source project. Model artifacts default below the ignored `artifacts/`
directory. Synthetic generators remain optional utilities and are not required
by the training workflow.

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

`data/requests/manifest.yaml` defines the fixed train, validation, and test
splits. `train_roadmap_ppo.py` defaults to the versioned topology, flows,
requests, manifest, and station configuration, while allowing explicit
overrides. It must not contain defaults beginning with `datasets/`, `exps/`,
or `models/`.

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
