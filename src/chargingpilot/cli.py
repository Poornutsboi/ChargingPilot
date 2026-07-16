from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from chargingpilot.environment.data_factory import DataFactory, DataFactoryConfig
from chargingpilot.environment.models import EpisodeData, SplitChargingEnvConfig, VehicleRequest
from chargingpilot.environment.split_charging_env import SplitChargingRequestEnv
from chargingpilot.simulator.models import StationSpec, VehicleSpec
from chargingpilot.trainer.ppo_trainer import PPOTrainer, PPOTrainerConfig


DEFAULT_DATA_DIR = Path("datasets/train")
DEFAULT_STATION_SETTING = Path("exps/data/setting_7stations_pv_ess.yaml")
DEFAULT_TRAVEL_TIME_MODEL = Path("models/highway_travel_time_gct.pt")
DEFAULT_OUTPUT = Path("models/ppo_split_charging.pt")
DEFAULT_CHECKPOINT_INTERVAL = 5000
DEFAULT_REQUEST_DIR = Path("datasets/request")
DEFAULT_REQUEST_MANIFEST = Path("exps/data/request_split.yaml")
DEFAULT_HIERARCHICAL_STATION_SETTING = Path(
    "exps/data/setting_72stations_roadmap_pv_ess.yaml"
)
DEFAULT_ROAD_MAP_NODES = Path("datasets/road_map/nodes_final.geojson")
DEFAULT_ROAD_MAP_LINKS = Path("datasets/road_map/links_final.geojson")
DEFAULT_LINK_FLOWS = Path("datasets/road_map/link_24h_flows.csv")
DEFAULT_ROUTE_CACHE = Path("datasets/road_map/distance_oracle_cache.json")
DEFAULT_HIERARCHICAL_OUTPUT = Path("models/hierarchical_ppo.pt")
DEFAULT_SELECTION_RECORD = Path("models/hierarchical_ppo_selected.txt")
# Acceptance contract: at most this many complete rollouts is a short smoke run.
HIERARCHICAL_SMOKE_MAX_FULL_ROLLOUTS = 2


@dataclass(frozen=True)
class PPOTrainingResult:
    checkpoint_path: Path
    metrics: dict[str, float]


@dataclass(frozen=True)
class ValidationCheckpoint:
    path: Path
    mean_wait: float
    p95_wait: float
    feasible: bool


@dataclass(frozen=True)
class HierarchicalTrainingInputs:
    factory_config: DataFactoryConfig
    oracle: object
    generator: object
    env: object


class _ZeroNetwork:
    def path_time(self, u: int, v: int, t: float, route_nodes=None) -> float:
        return 0.0

    def path_energy(self, u: int, v: int, t: float, vehicle_or_rho, route_nodes=None) -> float:
        return 0.0


def _demo_episode_factory() -> EpisodeData:
    station = StationSpec(
        station_id=1,
        charge_capacity=2,
        p_plug_kw=120.0,
        p_max_kw=120.0,
        eta=1.0,
    )
    vehicles = []
    for vehicle_id in range(1, 9):
        spec = VehicleSpec(
            battery_capacity=60.0,
            initial_soc=0.35,
            soc_min=0.0,
            p_max_kw=120.0,
            p_min_kw=30.0,
            rho_kwh_per_km=0.18,
            origin=1,
            destination=1,
            departure_time=float(vehicle_id - 1),
            path_nodes=(1,),
            path_edges=(),
            candidate_stations=(1,),
        )
        vehicles.append(
            VehicleRequest(
                vehicle_id=vehicle_id,
                decision_time=float(vehicle_id - 1),
                vehicle_spec=spec,
                target_soc=0.7,
            )
        )
    return EpisodeData(
        station_specs=(station,),
        vehicle_requests=tuple(vehicles),
        network=_ZeroNetwork(),
        timestep_minutes=1.0,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="Train MASTER-style PPO for split charging.")
    data_source = parser.add_mutually_exclusive_group()
    data_source.add_argument("--use-data-factory", dest="use_data_factory", action="store_true", default=True)
    data_source.add_argument("--demo", dest="use_data_factory", action="store_false")
    parser.add_argument("--hierarchical", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--station-setting",
        type=Path,
        default=DEFAULT_STATION_SETTING,
    )
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--travel-time-model", dest="travel_time_model", type=Path, default=DEFAULT_TRAVEL_TIME_MODEL)
    parser.add_argument("--no-travel-time-model", dest="travel_time_model", action="store_const", const=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--total-updates", type=int, default=10)
    parser.add_argument("--episodes-per-update", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--use-popart", dest="use_popart", action="store_true", default=True)
    parser.add_argument("--no-popart", dest="use_popart", action="store_false")
    parser.add_argument("--use-swanlab", dest="use_swanlab", action="store_true", default=True)
    parser.add_argument("--no-swanlab", dest="use_swanlab", action="store_false")
    parser.add_argument("--project", default="renewable-aware-split-charging")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--api-key", dest="api_key", default=None)
    parser.add_argument("--swanlab-mode", default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL)
    parser.add_argument(
        "--request-manifest",
        "--data-manifest",
        dest="request_manifest",
        type=Path,
        default=DEFAULT_REQUEST_MANIFEST,
    )
    parser.add_argument(
        "--request-split", choices=("train", "validation", "test"), default="train"
    )
    parser.add_argument("--road-map-nodes", type=Path, default=DEFAULT_ROAD_MAP_NODES)
    parser.add_argument("--road-map-links", type=Path, default=DEFAULT_ROAD_MAP_LINKS)
    parser.add_argument("--link-flows", type=Path, default=DEFAULT_LINK_FLOWS)
    parser.add_argument("--route-cache", type=Path, default=DEFAULT_ROUTE_CACHE)
    parser.add_argument("--detour-limit", type=float, default=0.60)
    parser.add_argument("--rollout-steps", type=int, default=4096)
    parser.add_argument(
        "--total-decisions",
        "--total-environment-steps",
        dest="total_decisions",
        type=int,
        default=1_000_000,
    )
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--selection-record", type=Path, default=DEFAULT_SELECTION_RECORD)
    parser.add_argument("--post-selection-test", action="store_true")
    parser.add_argument("--preflight-decisions", type=int, default=32)
    parser.add_argument(
        "--formal-short-run",
        action="store_true",
        help=(
            "force all-manifest preflight and full validation for a short run; "
            "by the acceptance contract, at most "
            f"{HIERARCHICAL_SMOKE_MAX_FULL_ROLLOUTS} full rollouts otherwise "
            "infer smoke mode"
        ),
    )
    args = parser.parse_args(raw_argv)
    if args.hierarchical or args.post_selection_test:
        if args.data_dir == DEFAULT_DATA_DIR:
            args.data_dir = DEFAULT_REQUEST_DIR
        if args.station_setting == DEFAULT_STATION_SETTING:
            args.station_setting = DEFAULT_HIERARCHICAL_STATION_SETTING
        if args.output == DEFAULT_OUTPUT:
            args.output = DEFAULT_HIERARCHICAL_OUTPUT
        if "--batch-size" not in raw_argv:
            args.batch_size = 256
        if "--learning-rate" not in raw_argv:
            args.learning_rate = 1e-4
        if "--use-popart" not in raw_argv and "--no-popart" not in raw_argv:
            args.use_popart = False
    if args.hierarchical and not args.post_selection_test and args.request_split != "train":
        parser.error("hierarchical training requires --request-split train")
    return args


def select_validation_checkpoint(
    candidates: Sequence[ValidationCheckpoint],
) -> ValidationCheckpoint:
    feasible = [candidate for candidate in candidates if bool(candidate.feasible)]
    if not feasible:
        raise RuntimeError("no feasible validation checkpoint is available")
    return min(
        feasible,
        key=lambda candidate: (
            float(candidate.mean_wait),
            float(candidate.p95_wait),
            str(candidate.path),
        ),
    )


def record_selected_checkpoint(record_path: Path, checkpoint_path: Path) -> Path:
    record_path = Path(record_path)
    selected = Path(checkpoint_path).resolve()
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(f"{selected}\n", encoding="utf-8")
    return selected


def require_split_access(
    split: str,
    *,
    selection_record: Path,
    post_selection: bool,
) -> Path | None:
    if split != "test":
        return None
    if not post_selection:
        raise RuntimeError("test split evaluation is forbidden before checkpoint selection")
    record_path = Path(selection_record)
    if not record_path.is_file():
        raise RuntimeError("test split evaluation requires a recorded selected checkpoint")
    value = record_path.read_text(encoding="utf-8").strip()
    if not value:
        raise RuntimeError("selected checkpoint record is empty")
    checkpoint = Path(value)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"recorded selected checkpoint does not exist: {checkpoint}")
    return checkpoint.resolve()


def require_training_split(args: argparse.Namespace) -> None:
    if str(args.request_split) != "train":
        raise ValueError("hierarchical training requires request_split='train'")


def completed_vehicle_waits(env: object) -> tuple[float, ...]:
    if getattr(env, "current_vehicle_id", None) is not None:
        raise RuntimeError("completed vehicle waits are available only after terminal drain")
    simulator = getattr(env, "simulator", None)
    history_log = getattr(simulator, "history_log", None)
    if history_log is None or not callable(getattr(history_log, "records", None)):
        raise RuntimeError("environment does not expose completed charging history")
    waits_by_vehicle: dict[int, float] = {}
    for record in history_log.records():
        vehicle_id = int(record.vehicle_id)
        wait_time = float(record.wait_time)
        if not np.isfinite(wait_time) or wait_time < 0.0:
            raise RuntimeError(
                f"completed charging history has invalid wait for vehicle={vehicle_id}"
            )
        waits_by_vehicle[vehicle_id] = waits_by_vehicle.get(vehicle_id, 0.0) + wait_time
    return tuple(waits_by_vehicle[key] for key in sorted(waits_by_vehicle))


def summarize_vehicle_waits(waits: Sequence[float]) -> tuple[float, float]:
    values = np.asarray(tuple(float(value) for value in waits), dtype=np.float64)
    if values.size == 0:
        return float("inf"), float("inf")
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("vehicle waits must be finite and non-negative")
    return float(values.sum() / values.size), float(np.percentile(values, 95))


def build_hierarchical_inputs(
    args: argparse.Namespace, *, for_training: bool = True
) -> HierarchicalTrainingInputs:
    from chargingpilot.environment.hierarchical_split_charging_env import HierarchicalSplitChargingRequestEnv
    from chargingpilot.environment.models import HierarchicalSplitChargingEnvConfig
    from chargingpilot.routing.distance_oracle import RoadDistanceOracle, _load_shortest_path_module
    from chargingpilot.routing.feasible_plan_generator import FeasiblePlanGenerator

    if for_training:
        require_training_split(args)
    require_split_access(
        str(args.request_split),
        selection_record=Path(args.selection_record),
        post_selection=bool(args.post_selection_test),
    )
    for path, label in (
        (args.data_dir, "data_dir"),
        (args.request_manifest, "request_manifest"),
        (args.station_setting, "station_setting"),
        (args.road_map_nodes, "road_map_nodes"),
        (args.road_map_links, "road_map_links"),
        (args.link_flows, "link_flows"),
    ):
        if label == "data_dir":
            _require_directory(Path(path), label)
        else:
            _require_file(Path(path), label)
    factory_config = _hierarchical_factory_config(args, str(args.request_split))
    factory = DataFactory(factory_config)
    shortest_path = _load_shortest_path_module()
    distance_map = shortest_path.load_road_map(
        Path(args.road_map_nodes),
        Path(args.road_map_links),
        undirected=True,
        weight_mode="distance",
    )
    station_ids = tuple(int(spec.station_id) for spec in factory._station_specs)
    oracle = RoadDistanceOracle(
        distance_map,
        station_ids,
        directed=False,
        cache_path=Path(args.route_cache),
    )
    generator = FeasiblePlanGenerator(
        oracle, station_ids, detour_limit=float(args.detour_limit)
    )
    env = HierarchicalSplitChargingRequestEnv(
        episode_factory=factory,
        oracle=oracle,
        config=HierarchicalSplitChargingEnvConfig(
            max_detour_ratio=float(args.detour_limit)
        ),
        plan_generator=generator,
    )
    return HierarchicalTrainingInputs(factory_config, oracle, generator, env)


def _hierarchical_factory_config(
    args: argparse.Namespace, split: str
) -> DataFactoryConfig:
    return DataFactoryConfig(
        episodes_dir=Path(args.data_dir),
        station_setting_path=Path(args.station_setting),
        request_manifest_path=Path(args.request_manifest),
        request_split=split,
        seed=int(args.seed),
        shuffle=split == "train",
        use_vdf_road_map=True,
        road_map_nodes_path=Path(args.road_map_nodes),
        road_map_links_path=Path(args.road_map_links),
        link_flows_path=Path(args.link_flows),
        road_map_undirected=True,
    )


def _fresh_hierarchical_env(
    inputs: HierarchicalTrainingInputs, *, seed: int, split: str | None = None
):
    from chargingpilot.environment.hierarchical_split_charging_env import HierarchicalSplitChargingRequestEnv
    from chargingpilot.environment.models import HierarchicalSplitChargingEnvConfig
    from chargingpilot.routing.feasible_plan_generator import FeasiblePlanGenerator

    config = inputs.factory_config
    overrides = {"seed": int(seed)}
    if split is not None:
        overrides.update(request_split=split, shuffle=split == "train")
    config = DataFactoryConfig(**{**asdict(config), **overrides})
    factory = DataFactory(config)
    generator = FeasiblePlanGenerator(
        inputs.oracle,
        tuple(inputs.oracle.station_ids),
        detour_limit=float(inputs.env.config.max_detour_ratio),
    )
    return HierarchicalSplitChargingRequestEnv(
        episode_factory=factory,
        oracle=inputs.oracle,
        config=HierarchicalSplitChargingEnvConfig(
            max_detour_ratio=float(inputs.env.config.max_detour_ratio)
        ),
        plan_generator=generator,
    )


def preflight_hierarchical_requests(
    generator: object,
    requests: Sequence[object],
    *,
    detour_limit: float,
    episode_index: int | None = None,
) -> dict[str, int]:
    ordered_requests = sorted(
        requests,
        key=lambda item: (float(item.decision_time), int(item.vehicle_id)),
    )
    if not ordered_requests:
        raise RuntimeError("preflight found no requests")
    request_count = 0
    for request in ordered_requests:
        try:
            context = generator.build_request_context(request)
            baseline = context.baseline
            baseline_nodes = tuple(int(value) for value in baseline.node_ids)
            if (
                not np.isfinite(float(baseline.distance_m))
                or float(baseline.distance_m) <= 0.0
                or not baseline_nodes
                or baseline_nodes[0] != int(request.vehicle_spec.origin)
                or baseline_nodes[-1] != int(request.vehicle_spec.destination)
                or int(baseline.station_id) not in baseline_nodes
            ):
                raise ValueError("service baseline is unreachable or malformed")

            action = generator.find_first_feasible_action(context)
            if action is None:
                raise ValueError("no feasible charging action")
            plan = generator.materialize_plan(context, action)
            selected_stations = (int(plan.s1),) + (
                () if plan.s2 is None else (int(plan.s2),)
            )
            route_nodes = tuple(int(value) for value in plan.route.node_ids)
            if (
                not route_nodes
                or route_nodes[0] != int(request.vehicle_spec.origin)
                or route_nodes[-1] != int(request.vehicle_spec.destination)
                or tuple(int(value) for value in plan.route.required_station_ids)
                != selected_stations
                or any(value not in route_nodes for value in selected_stations)
                or not np.isfinite(float(plan.detour_ratio))
                or float(plan.detour_ratio) > float(detour_limit) + 1e-9
            ):
                raise ValueError("materialized plan failed route or detour validation")
        except Exception as exc:
            episode_label = (
                "" if episode_index is None else f"episode={episode_index}, "
            )
            raise RuntimeError(
                "preflight feasibility failed for "
                f"{episode_label}vehicle={int(request.vehicle_id)}: {exc}"
            ) from exc
        request_count += 1
    return {"requests": int(request_count), "feasible_requests": int(request_count)}


def preflight_hierarchical_training(
    inputs: HierarchicalTrainingInputs, args: argparse.Namespace
) -> dict[str, int]:
    preflight_env = _fresh_hierarchical_env(inputs, seed=int(args.seed))
    factory = preflight_env.episode_factory
    episode_paths = getattr(factory, "_episode_paths", None)
    episode_count = 1 if episode_paths is None else len(episode_paths)
    if episode_count <= 0:
        raise RuntimeError("preflight found no manifest episodes")
    episodes = [preflight_env._initial_episode]
    episodes.extend(factory() for _ in range(episode_count - 1))

    request_count = 0
    for episode_index, episode in enumerate(episodes):
        episode_stats = preflight_hierarchical_requests(
            preflight_env.generator,
            episode.vehicle_requests,
            detour_limit=float(args.detour_limit),
            episode_index=episode_index,
        )
        request_count += int(episode_stats["requests"])
    stats = {
        "nodes": len(inputs.oracle.node_ids),
        "stations": len(inputs.oracle.station_ids),
        "cached_routes": len(getattr(inputs.oracle, "_cache_records", {})),
        "episodes": int(episode_count),
        "requests": int(request_count),
        "feasible_requests": int(request_count),
    }
    print(
        "preflight: " + ", ".join(f"{key}={value}" for key, value in stats.items()),
        flush=True,
    )
    return stats


def preflight_hierarchical_smoke(
    inputs: HierarchicalTrainingInputs, args: argparse.Namespace
) -> dict[str, int]:
    requested = int(args.preflight_decisions)
    if requested <= 0:
        raise ValueError("--preflight-decisions must be positive")
    preflight_env = _fresh_hierarchical_env(inputs, seed=int(args.seed))
    requests = sorted(
        preflight_env._initial_episode.vehicle_requests,
        key=lambda item: (float(item.decision_time), int(item.vehicle_id)),
    )[:requested]
    request_stats = preflight_hierarchical_requests(
        preflight_env.generator,
        requests,
        detour_limit=float(args.detour_limit),
        episode_index=0,
    )
    stats = {
        "nodes": len(inputs.oracle.node_ids),
        "stations": len(inputs.oracle.station_ids),
        "cached_routes": len(getattr(inputs.oracle, "_cache_records", {})),
        "episodes": 1,
        **request_stats,
    }
    print(
        "preflight-smoke: "
        + ", ".join(f"{key}={value}" for key, value in stats.items()),
        flush=True,
    )
    return stats


def _feasible_actions(env: object) -> list[tuple[object, float, bool]]:
    from chargingpilot.routing.models import HierarchicalAction

    actions: list[tuple[object, float, bool]] = []
    s1_context = env.s1_context()
    for s1_index in np.flatnonzero(s1_context.mask):
        s2_context = env.s2_context(int(s1_index))
        for s2_index in np.flatnonzero(s2_context.mask):
            route = s2_context.routes[int(s2_index)]
            if route is None:
                continue
            if int(s2_index) == len(env.oracle.station_ids):
                actions.append(
                    (HierarchicalAction(int(s1_index), int(s2_index), None), float(route.distance_m), False)
                )
                continue
            lambdas = env.lambda_context(int(s1_index), int(s2_index))
            if lambdas is None:
                continue
            for lambda_index in np.flatnonzero(lambdas.mask):
                actions.append(
                    (
                        HierarchicalAction(int(s1_index), int(s2_index), int(lambda_index)),
                        float(route.distance_m),
                        True,
                    )
                )
    if not actions:
        raise RuntimeError("rule-policy preflight found no feasible hierarchical action")
    return actions


def _rule_action(env: object, observation: object, policy_name: str, rng: random.Random):
    candidates = _feasible_actions(env)
    if policy_name == "random_feasible":
        return candidates[rng.randrange(len(candidates))][0]
    singles = [item for item in candidates if not item[2]]
    if policy_name == "mandatory_service_shortest":
        baseline_station = int(env.current_request_context.baseline.station_id)
        preferred = [
            item
            for item in singles
            if env.oracle.station_ids[item[0].s1_index] == baseline_station
        ]
        if preferred:
            return min(preferred, key=lambda item: item[0].s1_index)[0]
    if policy_name == "minimum_wait_single" and singles:
        return min(
            singles,
            key=lambda item: (
                float(observation.stations[item[0].s1_index, 16]),
                item[0].s1_index,
            ),
        )[0]
    return min(
        candidates,
        key=lambda item: (
            item[1],
            item[2],
            item[0].s1_index,
            item[0].s2_index,
            -1 if item[0].lambda_index is None else item[0].lambda_index,
        ),
    )[0]


def rule_plan_feasibility(
    env: object,
    request: object,
    plan: object,
    *,
    detour_limit: float,
) -> dict[str, bool]:
    route = plan.route
    node_ids = tuple(int(node_id) for node_id in route.node_ids)
    selected_stations = (int(plan.s1),) + (() if plan.s2 is None else (int(plan.s2),))
    required_stations = tuple(int(value) for value in route.required_station_ids)
    route_feasible = bool(
        node_ids
        and node_ids[0] == int(request.vehicle_spec.origin)
        and node_ids[-1] == int(request.vehicle_spec.destination)
        and required_stations == selected_stations
        and all(station_id in node_ids for station_id in selected_stations)
    )
    try:
        env.generator.validate_plan(request, plan)
    except RuntimeError:
        soc_feasible = False
    else:
        soc_feasible = True
    detour_feasible = bool(
        np.isfinite(float(plan.detour_ratio))
        and float(plan.detour_ratio) <= float(detour_limit) + 1e-9
    )
    return {
        "route": route_feasible,
        "soc": soc_feasible,
        "detour": detour_feasible,
    }


def run_rule_policy_checks(
    inputs: HierarchicalTrainingInputs, args: argparse.Namespace
) -> dict[str, dict[str, float]]:
    requested = int(args.preflight_decisions)
    if requested <= 0:
        raise ValueError("--preflight-decisions must be positive")
    results: dict[str, dict[str, float]] = {}
    for offset, policy_name in enumerate(
        ("mandatory_service_shortest", "minimum_wait_single", "random_feasible")
    ):
        env = _fresh_hierarchical_env(inputs, seed=int(args.seed) + offset)
        rng = random.Random(int(args.seed) + offset)
        observation, _ = env.reset(seed=int(args.seed) + offset)
        decisions = empty_masks = 0
        feasibility_counts = {"route": 0, "soc": 0, "detour": 0}
        while decisions < requested:
            if not np.isfinite(observation.request).all() or not np.isfinite(
                observation.stations
            ).all():
                raise RuntimeError(f"{policy_name} produced a non-finite observation")
            if not env.s1_context().mask.any():
                empty_masks += 1
                break
            action = _rule_action(env, observation, policy_name, rng)
            context = env.current_request_context
            request = context.request
            materialized_plan = env.generator.materialize_plan(context, action)
            plan_checks = rule_plan_feasibility(
                env,
                request,
                materialized_plan,
                detour_limit=float(args.detour_limit),
            )
            for name, passed in plan_checks.items():
                feasibility_counts[name] += int(passed)
            observation, reward, terminated, truncated, info = env.step(action)
            if not np.isfinite(reward):
                raise RuntimeError(f"{policy_name} produced a non-finite reward")
            if info["plan"] != materialized_plan:
                feasibility_counts["route"] -= 1
            decisions += 1
            if terminated or truncated:
                observation, _ = env.reset()
        if empty_masks or any(count != decisions for count in feasibility_counts.values()):
            raise RuntimeError(
                f"rule-policy preflight failed for {policy_name}: decisions={decisions}, "
                f"empty_masks={empty_masks}, feasibility={feasibility_counts}"
            )
        results[policy_name] = {
            "decisions": float(decisions),
            "empty_masks": 0.0,
            "route_feasibility": 1.0,
            "soc_feasibility": 1.0,
            "detour_feasibility": 1.0,
            "route_soc_detour_feasibility": 1.0,
        }
        print(
            f"rule-check {policy_name}: decisions={decisions}, empty_masks=0, "
            "route=100%, soc=100%, detour=100%",
            flush=True,
        )
    return results


def build_hierarchical_trainer(
    inputs: HierarchicalTrainingInputs, args: argparse.Namespace
):
    from chargingpilot.trainer.hierarchical_ppo_trainer import HierarchicalPPOConfig, HierarchicalPPOTrainer

    env = _fresh_hierarchical_env(inputs, seed=int(args.seed))
    trainer = HierarchicalPPOTrainer(
        env=env,
        config=HierarchicalPPOConfig(
            rollout_steps=int(args.rollout_steps),
            minibatch_size=int(args.batch_size),
            update_epochs=int(args.update_epochs),
            learning_rate=float(args.learning_rate),
            seed=int(args.seed),
            use_popart=bool(args.use_popart),
        ),
    )
    if args.resume is not None:
        trainer.load_checkpoint(Path(args.resume), resume=True)
    return trainer


def validate_hierarchical_checkpoint(
    inputs: HierarchicalTrainingInputs,
    args: argparse.Namespace,
    checkpoint_path: Path,
    *,
    split: str = "validation",
) -> ValidationCheckpoint:
    from chargingpilot.environment.request_manifest import RequestManifest
    from chargingpilot.trainer.hierarchical_policy import HierarchicalActorCritic

    require_split_access(
        split,
        selection_record=Path(args.selection_record),
        post_selection=bool(args.post_selection_test),
    )
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    station_indices = tuple(
        int(inputs.oracle.node_to_index[station_id])
        for station_id in inputs.oracle.station_ids
    )
    policy = HierarchicalActorCritic(
        node_count=len(inputs.oracle.node_ids),
        station_node_indices=station_indices,
    )
    state = checkpoint.get("model_state_dict", checkpoint.get("policy_state_dict"))
    if state is None:
        raise ValueError(f"checkpoint has no policy state: {checkpoint_path}")
    policy.load_state_dict(state)
    policy.eval()
    env = _fresh_hierarchical_env(inputs, seed=int(args.seed), split=split)
    manifest = RequestManifest.load(Path(args.request_manifest), Path(args.data_dir))
    episode_count = len(getattr(manifest, split))
    waits: list[float] = []
    feasible = True
    with torch.no_grad():
        for episode_index in range(episode_count):
            observation, _ = env.reset(seed=int(args.seed) + episode_index)
            terminated = truncated = False
            while not (terminated or truncated):
                s1 = env.s1_context()
                if not s1.mask.any():
                    feasible = False
                    break
                sample = policy.sample_action(
                    observation, s1, env.generator, deterministic=True
                )
                observation, reward, terminated, truncated, info = env.step(sample.action)
                if not np.isfinite(reward) or not np.isfinite(observation.request).all() or not np.isfinite(observation.stations).all():
                    feasible = False
                    break
                plan = info["plan"]
                if float(plan.detour_ratio) > float(args.detour_limit) + 1e-9:
                    feasible = False
                    break
            if not feasible:
                break
            waits.extend(completed_vehicle_waits(env))
    mean_wait, p95_wait = summarize_vehicle_waits(waits)
    return ValidationCheckpoint(Path(checkpoint_path), mean_wait, p95_wait, feasible)


def run_post_selection_test(args: argparse.Namespace) -> PPOTrainingResult:
    checkpoint = require_split_access(
        "test",
        selection_record=Path(args.selection_record),
        post_selection=True,
    )
    assert checkpoint is not None
    test_args = argparse.Namespace(**{**vars(args), "request_split": "test", "post_selection_test": True})
    inputs = build_hierarchical_inputs(test_args, for_training=False)
    result = validate_hierarchical_checkpoint(
        inputs, test_args, checkpoint, split="test"
    )
    if not result.feasible:
        raise RuntimeError("selected checkpoint failed post-selection test feasibility")
    return PPOTrainingResult(
        checkpoint_path=checkpoint,
        metrics={"test_mean_wait": result.mean_wait, "test_p95_wait": result.p95_wait},
    )


def is_hierarchical_training_smoke(
    total_decisions: int, rollout_steps: int
) -> bool:
    decisions = int(total_decisions)
    rollout = int(rollout_steps)
    if rollout <= 0:
        raise ValueError("rollout_steps must be positive")
    return 0 < decisions <= HIERARCHICAL_SMOKE_MAX_FULL_ROLLOUTS * rollout


def run_hierarchical_training(args: argparse.Namespace) -> PPOTrainingResult:
    require_training_split(args)
    inputs = build_hierarchical_inputs(args)
    budget = int(args.total_decisions)
    if budget < 0:
        raise ValueError("--total-decisions must be non-negative")
    rollout_steps = int(args.rollout_steps)
    smoke_run = (
        is_hierarchical_training_smoke(budget, rollout_steps)
        and not bool(getattr(args, "formal_short_run", False))
    )
    if smoke_run:
        smoke_limit = HIERARCHICAL_SMOKE_MAX_FULL_ROLLOUTS * rollout_steps
        print(
            "short-run smoke inferred by acceptance contract: "
            f"decisions={budget}, limit={smoke_limit} "
            f"({HIERARCHICAL_SMOKE_MAX_FULL_ROLLOUTS} full rollouts); "
            "use --formal-short-run to force all-manifest preflight and full validation",
            flush=True,
        )
        preflight_hierarchical_smoke(inputs, args)
    else:
        preflight_hierarchical_training(inputs, args)
    run_rule_policy_checks(inputs, args)
    trainer = build_hierarchical_trainer(inputs, args)
    if smoke_run:
        print(
            "validation deferred for smoke run: "
            f"decisions={budget}, rollout_steps={rollout_steps}",
            flush=True,
        )
    metrics: dict[str, float] = {}
    candidates: list[ValidationCheckpoint] = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    while budget > 0 and trainer.completed_decisions < budget:
        rollout = trainer.collect_rollout()
        metrics = dict(trainer.update(rollout))
        if not all(np.isfinite(float(value)) for value in metrics.values()):
            raise RuntimeError("hierarchical training produced non-finite update metrics")
        print(
            _format_training_metrics(metrics, trainer.completed_updates),
            flush=True,
        )
        if not smoke_run:
            checkpoint = _checkpoint_path_for_update(
                output, trainer.completed_updates
            )
            trainer.save_checkpoint(checkpoint, resumable=True)
            candidates.append(
                validate_hierarchical_checkpoint(inputs, args, checkpoint)
            )
    if budget > 0 and trainer.completed_decisions:
        trainer.save_checkpoint(output, resumable=True)
    completed = int(trainer.completed_decisions) if budget > 0 else 0
    metrics.update(
        decision_budget=float(budget),
        completed_decisions=float(completed),
        overshoot_decisions=float(max(0, completed - budget)),
    )
    if smoke_run:
        metrics["validation_deferred"] = 1.0
        return PPOTrainingResult(checkpoint_path=output, metrics=metrics)
    if candidates:
        selected = select_validation_checkpoint(candidates)
        record_selected_checkpoint(Path(args.selection_record), selected.path)
        metrics.update(
            validation_mean_wait=float(selected.mean_wait),
            validation_p95_wait=float(selected.p95_wait),
        )
        return PPOTrainingResult(checkpoint_path=selected.path, metrics=metrics)
    return PPOTrainingResult(checkpoint_path=output, metrics=metrics)


def run_training(args: argparse.Namespace) -> PPOTrainingResult:
    _set_seeds(int(args.seed))
    if bool(args.use_data_factory):
        _require_directory(Path(args.data_dir), "data_dir")
        _require_file(Path(args.station_setting), "station_setting")
        if args.travel_time_model is not None:
            _require_file(Path(args.travel_time_model), "travel_time_model")
        episode_factory = DataFactory(
            DataFactoryConfig(
                episodes_dir=args.data_dir,
                station_setting_path=args.station_setting,
                seed=args.seed if args.data_seed is None else args.data_seed,
                travel_time_model_path=args.travel_time_model,
            )
        )
        env_config = episode_factory.env_config()
    else:
        episode_factory = _demo_episode_factory
        env_config = SplitChargingEnvConfig(max_station_count=1)
    env = SplitChargingRequestEnv(episode_factory=episode_factory, config=env_config)
    env.reset(seed=int(args.seed))
    trainer_config = PPOTrainerConfig(
        total_updates=int(args.total_updates),
        episodes_per_update=int(args.episodes_per_update),
        batch_size=int(args.batch_size),
        update_epochs=int(args.update_epochs),
        learning_rate=float(args.learning_rate),
        use_popart=bool(args.use_popart),
    )
    trainer = PPOTrainer(env=env, config=trainer_config)
    swanlab_run = _init_swanlab(args, trainer_config)
    checkpoint_interval = int(args.checkpoint_interval)
    if checkpoint_interval < 0:
        raise ValueError("--checkpoint-interval must be non-negative")

    def on_update(item: dict[str, float], step: int) -> None:
        print(_format_training_metrics(item, step), flush=True)
        if swanlab_run is not None:
            swanlab_run.log(item, step=step)
        if checkpoint_interval > 0 and step % checkpoint_interval == 0:
            checkpoint_path = _checkpoint_path_for_update(Path(args.output), step)
            _save_checkpoint(trainer, checkpoint_path, item)
            print(f"checkpoint update {step}: {checkpoint_path}", flush=True)

    try:
        metrics = trainer.train(on_update=on_update)
    finally:
        if swanlab_run is not None:
            swanlab_run.finish()
    checkpoint_path = _save_checkpoint(trainer, Path(args.output), metrics)
    return PPOTrainingResult(checkpoint_path=checkpoint_path, metrics=dict(metrics))


def main() -> None:
    print("Start Training", flush=True)
    args = parse_args()
    result = (
        run_post_selection_test(args)
        if bool(args.post_selection_test)
        else run_hierarchical_training(args)
        if bool(args.hierarchical)
        else run_training(args)
    )
    print("Training Finished", flush=True)
    print(result.metrics)
    print(f"checkpoint: {result.checkpoint_path}")


def _set_seeds(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _save_checkpoint(
    trainer: PPOTrainer,
    output_path: Path,
    metrics: dict[str, float],
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_state_dict": trainer.policy.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "trainer_config": asdict(trainer.config),
            "metrics": dict(metrics),
        },
        output_path,
    )
    return output_path


def _checkpoint_path_for_update(output_path: Path, update: int) -> Path:
    suffix = output_path.suffix or ".pt"
    stem = output_path.stem or "checkpoint"
    return output_path.with_name(f"{stem}_update_{int(update):06d}{suffix}")


def _format_training_metrics(metrics: dict[str, float], step: int) -> str:
    preferred_order = (
        "updates",
        "rollout_size",
        "policy_loss",
        "value_loss",
        "entropy",
        "clip_fraction",
        "mean_reward",
        "popart_mean",
        "popart_std",
    )
    keys = [key for key in preferred_order if key in metrics]
    keys.extend(sorted(key for key in metrics if key not in preferred_order))
    formatted = ", ".join(f"{key}={_format_metric_value(metrics[key])}" for key in keys)
    return f"training update {int(step)}: {formatted}"


def _format_metric_value(value: float) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _init_swanlab(args: argparse.Namespace, trainer_config: PPOTrainerConfig):
    if not bool(args.use_swanlab):
        return None
    import swanlab

    init_kwargs = {
        "project": str(args.project),
        "experiment_name": args.run_name,
        "config": {
            "trainer_config": asdict(trainer_config),
            "data_dir": str(args.data_dir),
            "station_setting": str(args.station_setting),
            "travel_time_model": None if args.travel_time_model is None else str(args.travel_time_model),
            "output": str(args.output),
            "seed": int(args.seed),
            "data_seed": args.data_seed,
            "use_data_factory": bool(args.use_data_factory),
            "use_popart": bool(args.use_popart),
            "online": bool(args.online),
            "api_key_configured": args.api_key is not None,
            "checkpoint_interval": int(args.checkpoint_interval),
        },
    }
    if args.api_key is not None:
        swanlab.login(api_key=str(args.api_key), save=False)
    mode = "cloud" if bool(args.online) else args.swanlab_mode
    if mode is not None:
        init_kwargs["mode"] = str(mode)
    swanlab.init(**init_kwargs)
    return swanlab


def _require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist or is not a directory: {path}")


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {path}")


if __name__ == "__main__":
    main()
