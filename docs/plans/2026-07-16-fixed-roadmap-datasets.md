# Fixed RoadMap Dataset Migration Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train RoadMap PPO directly from versioned request and traffic-flow datasets.

**Architecture:** Copy the fixed 20 request CSV files, their existing split manifest, and the generated traffic-flow CSV into `data/`. Make the standalone PPO entry point default to these files; keep synthetic generators optional.

---

### Task 1: Version fixed RoadMap training inputs

- [ ] Write a failing test that loads `data/requests/manifest.yaml` through `RequestManifest` and verifies each split references a tracked file.
- [ ] Copy `datasets/request/*.csv`, `exps/data/request_split.yaml`, and `datasets/road_map/link_24h_flows.csv` into the standalone data layout.
- [ ] Run the dataset test and verify it passes.

### Task 2: Default training to the fixed dataset

- [ ] Write a failing CLI-argument test proving `train_roadmap_ppo.py` defaults to `data/requests`, `data/requests/manifest.yaml`, and `data/roadmap/link_24h_flows.csv`.
- [ ] Update the entry point and README; preserve explicit overrides.
- [ ] Run focused CLI and package tests.

### Task 3: Verify and publish

- [ ] Run `python -m pytest -q`.
- [ ] Run a short PPO smoke command using only defaults plus an ignored `--output-dir`.
- [ ] Audit staged files to ensure the intended fixed data is tracked and no models, logs, or generated artifacts are tracked.
- [ ] Commit and push `main`.
