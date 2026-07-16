# Standalone RoadMap PPO Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and save a RoadMap charging-allocation PPO model from a clean ChargingPilot checkout.

**Architecture:** Replace copied legacy CLI code with focused package APIs and explicit standalone scripts. Generated flows, request CSVs, manifests, route caches, and checkpoints live below a caller-selected output directory; only topology and configuration are versioned.

**Tech Stack:** Python, PyTorch, Gymnasium, PyYAML, pytest.

---

### Task 1: Establish standalone path and generation APIs

**Files:** Create `chargingpilot/paths.py`, `roadmap/manifest.py`, `scripts/build_manifest.py`; modify road-map generators and tests.

- [ ] Write failing tests that request defaults resolve to `data/roadmap` and that generated request CSVs produce a valid train/validation/test manifest.
- [ ] Run the tests and observe missing module or incorrect legacy-path failures.
- [ ] Implement package path helpers, remove generator defaults that point inside source directories, and implement deterministic manifest splitting.
- [ ] Run focused generator/manifest tests; verify they pass.
- [ ] Commit `feat: add standalone roadmap data pipeline`.

### Task 2: Remove legacy runtime coupling

**Files:** Modify `environment/data_factory.py`, `routing/distance_oracle.py`, `simulator/__init__.py`; delete or exclude `simulator/{orchestrator,reachability}.py`; add import-boundary tests.

- [ ] Write a failing test that imports every retained runtime package from an isolated checkout and asserts no dependency on `data.network`, `datasets/`, `exps/`, or `models/`.
- [ ] Run it and observe legacy reference failures.
- [ ] Replace dynamic source loading with package imports; make configuration and generated files explicit; remove unused legacy simulator modules.
- [ ] Run retained package and environment tests; verify all pass.
- [ ] Commit `refactor: remove legacy project coupling`.

### Task 3: Build dedicated RoadMap PPO training entry point

**Files:** Create `scripts/train_roadmap_ppo.py`; modify `trainer/hierarchical_ppo_trainer.py` only if API extraction is needed; add CLI tests.

- [ ] Write failing CLI tests for explicit `--requests-dir`, `--manifest`, `--flows`, `--station-config`, and `--output-dir` arguments.
- [ ] Run tests and observe missing entry point.
- [ ] Implement the script using `DataFactory`, the RoadMap environment, and `HierarchicalPPOTrainer`; direct all route cache, checkpoint, and selection outputs below `--output-dir`.
- [ ] Run CLI tests and a minimal one-update training run; assert a checkpoint exists.
- [ ] Commit `feat: add standalone roadmap PPO trainer`.

### Task 4: Verify the complete independent workflow

**Files:** Create `tests/test_roadmap_ppo_e2e.py`; modify `README.md`.

- [ ] Write an end-to-end failing test that invokes flows, requests, manifest, and a tiny PPO training run in a temporary directory.
- [ ] Run it and observe the missing standalone behavior.
- [ ] Complete the smallest implementation necessary for a green test.
- [ ] Run `python -m pytest -q` and then run documented shell commands from a clean output directory.
- [ ] Audit `git ls-files` for no generated files, model weights, logs, or old-project paths.
- [ ] Commit `test: verify standalone roadmap PPO workflow` and push `main`.
