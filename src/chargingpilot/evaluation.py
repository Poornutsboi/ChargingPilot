from __future__ import annotations

import math
import random
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from chargingpilot.routing.models import HierarchicalAction


@dataclass(frozen=True)
class BaselineSelection:
    action: HierarchicalAction
    fallback_used: bool


@dataclass(frozen=True)
class DayBootstrapInterval:
    lower: float
    upper: float
    confidence: float
    days: int
    resamples: int
    seed: int
    unit: str = "day"


@dataclass(frozen=True)
class StationVisit:
    station_id: int
    leg_index: int
    queue_length_at_arrival: int
    wait_minutes: float
    grid_energy_kwh: float
    curtailment_kwh: float
    energy_delivered_kwh: float

    def __post_init__(self) -> None:
        if int(self.station_id) < 0:
            raise ValueError("station_id must be nonnegative")
        if int(self.leg_index) not in {1, 2}:
            raise ValueError("leg_index must be 1 or 2")
        if int(self.queue_length_at_arrival) < 0:
            raise ValueError("queue_length_at_arrival must be nonnegative")
        for name in (
            "wait_minutes",
            "grid_energy_kwh",
            "curtailment_kwh",
            "energy_delivered_kwh",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class HierarchicalEvaluationRecord:
    checkpoint: str
    split: str
    seed: int
    detour_limit: float
    policy: str
    vehicle_id: int
    day: str
    load_class: str
    hour: int
    s1_station_id: int
    s2_station_id: int | None
    visits: tuple[StationVisit, ...]
    detour_ratio: float
    fallback_used: bool
    service_feasible: bool
    path_feasible: bool
    soc_feasible: bool
    detour_feasible: bool
    empty_mask: bool

    def __post_init__(self) -> None:
        if not str(self.checkpoint):
            raise ValueError("checkpoint must not be empty")
        if not str(self.split):
            raise ValueError("split must not be empty")
        if not str(self.policy):
            raise ValueError("policy must not be empty")
        if not str(self.day):
            raise ValueError("day must not be empty")
        if not str(self.load_class):
            raise ValueError("load_class must not be empty")
        if not 0 <= int(self.hour) <= 23:
            raise ValueError("hour must be in [0, 23]")
        if int(self.s1_station_id) < 0:
            raise ValueError("s1_station_id must be nonnegative")
        expected_station_ids = (int(self.s1_station_id),) + (
            () if self.s2_station_id is None else (int(self.s2_station_id),)
        )
        if tuple(int(visit.station_id) for visit in self.visits) != expected_station_ids:
            raise ValueError("visits must match s1/s2 station order exactly")
        if tuple(int(visit.leg_index) for visit in self.visits) != tuple(
            range(1, len(self.visits) + 1)
        ):
            raise ValueError("visits must use consecutive leg indices")
        for name in ("detour_limit", "detour_ratio"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")

    @property
    def plan_type(self) -> str:
        return "single" if self.s2_station_id is None else "split"

    @property
    def wait_minutes(self) -> float:
        return float(sum(visit.wait_minutes for visit in self.visits))

    @property
    def station_queue_length(self) -> int:
        return max((int(visit.queue_length_at_arrival) for visit in self.visits), default=0)

    @property
    def grid_energy_kwh(self) -> float:
        return float(sum(visit.grid_energy_kwh for visit in self.visits))

    @property
    def curtailment_kwh(self) -> float:
        return float(sum(visit.curtailment_kwh for visit in self.visits))

    @property
    def additional_stop(self) -> bool:
        return self.s2_station_id is not None


@dataclass(frozen=True)
class HierarchicalEvaluationMetadata:
    checkpoint: str
    split: str
    seed: int
    detour_limit: float
    policy: str
    fallback_count: int
    fallback_rate: float
    request_count: int
    bootstrap_seed: int
    bootstrap_resamples: int

    @classmethod
    def from_records(
        cls,
        records: Iterable[HierarchicalEvaluationRecord],
        *,
        bootstrap_seed: int = 7,
        bootstrap_resamples: int = 10_000,
    ) -> "HierarchicalEvaluationMetadata":
        items = list(records)
        _validate_records(items)
        first = items[0]
        fallback_count = sum(bool(item.fallback_used) for item in items)
        return cls(
            checkpoint=str(first.checkpoint),
            split=str(first.split),
            seed=int(first.seed),
            detour_limit=float(first.detour_limit),
            policy=str(first.policy),
            fallback_count=int(fallback_count),
            fallback_rate=float(fallback_count / len(items)),
            request_count=len(items),
            bootstrap_seed=int(bootstrap_seed),
            bootstrap_resamples=int(bootstrap_resamples),
        )


@dataclass(frozen=True)
class EvaluationEpisodeSpec:
    day: str
    load_class: str
    environment_factory: Callable[[], Any]

    def __post_init__(self) -> None:
        if not str(self.day):
            raise ValueError("day must not be empty")
        if not str(self.load_class):
            raise ValueError("load_class must not be empty")
        if not callable(self.environment_factory):
            raise TypeError("environment_factory must be callable")


@dataclass(frozen=True)
class HierarchicalPolicyEvaluationReport:
    metadata: HierarchicalEvaluationMetadata
    records: tuple[HierarchicalEvaluationRecord, ...]
    aggregates: tuple[HierarchicalEvaluationAggregate, ...]


@dataclass(frozen=True)
class _DecisionTrace:
    request: Any
    plan: Any
    fallback_used: bool
    empty_mask: bool


@dataclass(frozen=True)
class HierarchicalEvaluationAggregate:
    group_dimension: str
    group_value: str
    request_count: int
    wait_minutes_total: float
    wait_minutes_mean: float
    wait_minutes_p50: float
    wait_minutes_p95: float
    wait_minutes_p99: float
    wait_minutes_max: float
    wait_over_15_rate: float
    wait_over_30_rate: float
    wait_over_60_rate: float
    max_station_queue: int
    grid_energy_kwh_total: float
    grid_energy_kwh_mean: float
    curtailment_kwh_total: float
    curtailment_kwh_mean: float
    detour_ratio_mean: float
    detour_ratio_p95: float
    detour_ratio_max: float
    additional_stop_rate: float
    fallback_count: int
    fallback_rate: float
    service_feasible_rate: float
    path_feasible_rate: float
    soc_feasible_rate: float
    detour_feasible_rate: float
    hard_feasible_rate: float
    empty_mask_count: int
    bootstrap_unit: str
    bootstrap_days: int
    bootstrap_seed: int
    bootstrap_resamples: int
    wait_minutes_mean_ci_lower: float
    wait_minutes_mean_ci_upper: float
    grid_energy_kwh_mean_ci_lower: float
    grid_energy_kwh_mean_ci_upper: float
    curtailment_kwh_mean_ci_lower: float
    curtailment_kwh_mean_ci_upper: float
    detour_ratio_mean_ci_lower: float
    detour_ratio_mean_ci_upper: float


_METADATA_FIELDS = (
    "checkpoint",
    "split",
    "seed",
    "detour_limit",
    "policy",
)
_AGGREGATE_FIELDS = tuple(HierarchicalEvaluationAggregate.__dataclass_fields__)
EVALUATION_CSV_FIELDS = _METADATA_FIELDS + _AGGREGATE_FIELDS
REQUEST_CSV_FIELDS = (
    "checkpoint",
    "split",
    "seed",
    "detour_limit",
    "policy",
    "vehicle_id",
    "day",
    "load_class",
    "hour",
    "s1_station_id",
    "s2_station_id",
    "plan_type",
    "wait_minutes_total",
    "grid_energy_kwh_total",
    "curtailment_kwh_total",
    "detour_ratio",
    "additional_stop",
    "fallback_used",
    "service_feasible",
    "path_feasible",
    "soc_feasible",
    "detour_feasible",
    "empty_mask",
    "visits_json",
)


@dataclass(frozen=True)
class _MetricRow:
    record: HierarchicalEvaluationRecord
    visit: StationVisit | None = None

    @property
    def wait_minutes(self) -> float:
        return self.record.wait_minutes if self.visit is None else float(self.visit.wait_minutes)

    @property
    def grid_energy_kwh(self) -> float:
        return self.record.grid_energy_kwh if self.visit is None else float(self.visit.grid_energy_kwh)

    @property
    def curtailment_kwh(self) -> float:
        return self.record.curtailment_kwh if self.visit is None else float(self.visit.curtailment_kwh)

    @property
    def station_queue_length(self) -> int:
        return (
            self.record.station_queue_length
            if self.visit is None
            else int(self.visit.queue_length_at_arrival)
        )


@dataclass(frozen=True)
class _FeasibleCandidate:
    action: HierarchicalAction
    route_distance_m: float
    is_split: bool


def select_mandatory_service_shortest(
    generator: Any,
    context: Any,
) -> BaselineSelection:
    candidates = _feasible_candidates(generator, context)
    baseline_station_id = int(context.baseline.station_id)
    preferred = [
        item
        for item in candidates
        if not item.is_split
        and int(generator.station_ids[item.action.s1_index]) == baseline_station_id
    ]
    if preferred:
        return BaselineSelection(
            action=min(preferred, key=_deterministic_candidate_key).action,
            fallback_used=False,
        )
    return BaselineSelection(
        action=min(candidates, key=_deterministic_candidate_key).action,
        fallback_used=True,
    )


def select_minimum_wait_single(
    generator: Any,
    context: Any,
    *,
    estimated_wait_minutes: Mapping[int, float],
) -> BaselineSelection:
    candidates = _feasible_candidates(generator, context)
    singles = [item for item in candidates if not item.is_split]
    if singles:
        for item in singles:
            station_id = int(generator.station_ids[item.action.s1_index])
            if station_id not in estimated_wait_minutes:
                raise ValueError(
                    f"estimated_wait_minutes is missing feasible station {station_id}"
                )
            wait = float(estimated_wait_minutes[station_id])
            if not math.isfinite(wait) or wait < 0.0:
                raise ValueError(
                    f"estimated wait for station {station_id} must be finite and nonnegative"
                )
        selected = min(
            singles,
            key=lambda item: (
                float(
                    estimated_wait_minutes[
                        int(generator.station_ids[item.action.s1_index])
                    ]
                ),
                int(item.action.s1_index),
            ),
        )
        return BaselineSelection(action=selected.action, fallback_used=False)
    splits = [item for item in candidates if item.is_split]
    if not splits:
        raise RuntimeError("feasibility contexts contain no feasible charging action")
    return BaselineSelection(
        action=min(splits, key=_deterministic_candidate_key).action,
        fallback_used=True,
    )


def select_random_feasible(
    generator: Any,
    context: Any,
    *,
    rng: random.Random,
) -> BaselineSelection:
    candidates = _feasible_candidates(generator, context)
    return BaselineSelection(
        action=candidates[rng.randrange(len(candidates))].action,
        fallback_used=False,
    )


def bootstrap_day_confidence_interval(
    day_summaries: Mapping[str, float],
    *,
    seed: int,
    resamples: int = 10_000,
    confidence: float = 0.95,
) -> DayBootstrapInterval:
    if not day_summaries:
        raise ValueError("day_summaries must contain at least one day")
    if int(resamples) <= 0:
        raise ValueError("resamples must be positive")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be between zero and one")
    values = np.asarray(
        [float(day_summaries[day]) for day in sorted(day_summaries)],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("day_summaries must contain only finite values")
    rng = np.random.default_rng(int(seed))
    sampled_indices = rng.integers(
        0,
        values.size,
        size=(int(resamples), values.size),
    )
    statistics = values[sampled_indices].mean(axis=1)
    tail = (1.0 - float(confidence)) / 2.0
    lower, upper = np.quantile(statistics, (tail, 1.0 - tail))
    return DayBootstrapInterval(
        lower=float(lower),
        upper=float(upper),
        confidence=float(confidence),
        days=int(values.size),
        resamples=int(resamples),
        seed=int(seed),
    )


EVALUATION_POLICY_LABELS = (
    "ppo",
    "mandatory_service_shortest",
    "minimum_wait_single",
    "random_feasible",
)


def run_hierarchical_evaluation(
    episodes: Sequence[EvaluationEpisodeSpec],
    *,
    policy: Any,
    checkpoint: str,
    split: str,
    seed: int,
    detour_limit: float,
) -> dict[str, HierarchicalPolicyEvaluationReport]:
    episode_specs = tuple(episodes)
    if not episode_specs:
        raise ValueError("episodes must contain at least one episode specification")
    if not str(checkpoint):
        raise ValueError("checkpoint must not be empty")
    if not str(split):
        raise ValueError("split must not be empty")
    if not math.isfinite(float(detour_limit)) or float(detour_limit) <= 0.0:
        raise ValueError("detour_limit must be positive and finite")
    eval_method = getattr(policy, "eval", None)
    if callable(eval_method):
        eval_method()

    reports: dict[str, HierarchicalPolicyEvaluationReport] = {}
    for label in EVALUATION_POLICY_LABELS:
        records: list[HierarchicalEvaluationRecord] = []
        for episode_index, episode_spec in enumerate(episode_specs):
            environment = episode_spec.environment_factory()
            episode_seed = int(seed) + int(episode_index)
            observation, reset_info = environment.reset(seed=episode_seed)
            traces: dict[int, _DecisionTrace] = {}
            rng = random.Random(episode_seed)
            terminated = False
            while not terminated:
                vehicle_id = environment.current_vehicle_id
                context = environment.current_request_context
                if vehicle_id is None or context is None:
                    raise RuntimeError("environment has no current request before termination")
                s1_context = environment.s1_context()
                empty_mask = not bool(np.asarray(s1_context.mask, dtype=np.bool_).any())
                if empty_mask:
                    raise RuntimeError(
                        f"empty S1 feasibility mask for vehicle={vehicle_id}"
                    )
                selection = _select_evaluation_action(
                    label=label,
                    policy=policy,
                    environment=environment,
                    observation=observation,
                    s1_context=s1_context,
                    rng=rng,
                )
                action = selection.action
                _require_selected_contexts_nonempty(environment, action)
                next_observation, _reward, terminated, truncated, info = environment.step(
                    action
                )
                if truncated:
                    raise RuntimeError(
                        f"evaluation episode truncated for policy={label}, vehicle={vehicle_id}"
                    )
                traces[int(vehicle_id)] = _DecisionTrace(
                    request=context.request,
                    plan=info["plan"],
                    fallback_used=bool(selection.fallback_used),
                    empty_mask=bool(empty_mask),
                )
                observation = next_observation
                reset_info = info
            del reset_info
            history = tuple(environment.simulator.history_log.records())
            records.extend(
                _records_from_completed_history(
                    traces=traces,
                    history=history,
                    generator=environment.generator,
                    station_ids=tuple(int(value) for value in environment.oracle.station_ids),
                    checkpoint=str(checkpoint),
                    split=str(split),
                    seed=int(seed),
                    detour_limit=float(detour_limit),
                    policy=label,
                    day=str(episode_spec.day),
                    load_class=str(episode_spec.load_class),
                )
            )
        metadata = HierarchicalEvaluationMetadata.from_records(
            records, bootstrap_seed=int(seed), bootstrap_resamples=10_000
        )
        reports[label] = HierarchicalPolicyEvaluationReport(
            metadata=metadata,
            records=tuple(records),
            aggregates=tuple(
                aggregate_hierarchical_evaluation(
                    records, bootstrap_seed=int(seed), bootstrap_resamples=10_000
                )
            ),
        )
    return reports


def _select_evaluation_action(
    *,
    label: str,
    policy: Any,
    environment: Any,
    observation: Any,
    s1_context: Any,
    rng: random.Random,
) -> BaselineSelection:
    context = environment.current_request_context
    if label == "ppo":
        sample = policy.sample_action(
            observation,
            s1_context,
            environment.generator,
            deterministic=True,
        )
        return BaselineSelection(action=sample.action, fallback_used=False)
    if label == "mandatory_service_shortest":
        return select_mandatory_service_shortest(environment.generator, context)
    if label == "minimum_wait_single":
        waits = {
            int(station_id): float(observation.stations[index, 16])
            for index, station_id in enumerate(environment.generator.station_ids)
        }
        return select_minimum_wait_single(
            environment.generator,
            context,
            estimated_wait_minutes=waits,
        )
    if label == "random_feasible":
        return select_random_feasible(environment.generator, context, rng=rng)
    raise ValueError(f"unsupported evaluation policy label {label!r}")


def _require_selected_contexts_nonempty(environment: Any, action: HierarchicalAction) -> None:
    s2_context = environment.s2_context(action.s1_index)
    if not bool(np.asarray(s2_context.mask, dtype=np.bool_).any()):
        raise RuntimeError("selected S1 has an empty S2 feasibility mask")
    if action.s2_index == len(environment.generator.station_ids):
        return
    lambda_context = environment.lambda_context(action.s1_index, action.s2_index)
    if lambda_context is None or not bool(
        np.asarray(lambda_context.mask, dtype=np.bool_).any()
    ):
        raise RuntimeError("selected split action has an empty lambda feasibility mask")


def _records_from_completed_history(
    *,
    traces: Mapping[int, _DecisionTrace],
    history: Sequence[Any],
    generator: Any,
    station_ids: tuple[int, ...],
    checkpoint: str,
    split: str,
    seed: int,
    detour_limit: float,
    policy: str,
    day: str,
    load_class: str,
) -> list[HierarchicalEvaluationRecord]:
    by_vehicle: dict[int, list[Any]] = {}
    for item in history:
        by_vehicle.setdefault(int(item.vehicle_id), []).append(item)
    records: list[HierarchicalEvaluationRecord] = []
    for vehicle_id, trace in traces.items():
        completed = sorted(
            by_vehicle.get(int(vehicle_id), ()),
            key=lambda item: (float(item.arrival_time), int(item.station_id)),
        )
        plan = trace.plan
        expected_stations = (int(plan.s1),) + (
            () if plan.s2 is None else (int(plan.s2),)
        )
        if tuple(int(item.station_id) for item in completed) != expected_stations:
            raise RuntimeError(
                f"completed visits do not match plan for vehicle={vehicle_id}"
            )
        visits = tuple(
            StationVisit(
                station_id=int(item.station_id),
                leg_index=index,
                queue_length_at_arrival=_queue_length_at_arrival(item, history),
                wait_minutes=float(item.wait_time),
                grid_energy_kwh=float(item.grid_used_kwh),
                curtailment_kwh=float(item.renewable_curtailed_kwh),
                energy_delivered_kwh=float(item.energy_delivered_kwh),
            )
            for index, item in enumerate(completed, start=1)
        )
        request = trace.request
        route_nodes = tuple(int(value) for value in plan.route.node_ids)
        selected = expected_stations
        path_feasible = bool(
            route_nodes
            and route_nodes[0] == int(request.vehicle_spec.origin)
            and route_nodes[-1] == int(request.vehicle_spec.destination)
            and tuple(int(value) for value in plan.route.required_station_ids) == selected
            and all(station_id in route_nodes for station_id in selected)
        )
        service_feasible = bool(
            any(int(node_id) in set(station_ids) for node_id in route_nodes)
        )
        detour_feasible = bool(
            math.isfinite(float(plan.detour_ratio))
            and float(plan.detour_ratio) <= float(detour_limit) + 1e-9
        )
        soc_feasible = _executed_soc_is_feasible(
            request=request,
            plan=plan,
            completed=completed,
            generator=generator,
        )
        records.append(
            HierarchicalEvaluationRecord(
                checkpoint=checkpoint,
                split=split,
                seed=int(seed),
                detour_limit=float(detour_limit),
                policy=policy,
                vehicle_id=int(vehicle_id),
                day=day,
                load_class=load_class,
                hour=int(float(request.decision_time) // 60.0) % 24,
                s1_station_id=int(plan.s1),
                s2_station_id=None if plan.s2 is None else int(plan.s2),
                visits=visits,
                detour_ratio=float(plan.detour_ratio),
                fallback_used=bool(trace.fallback_used),
                service_feasible=service_feasible,
                path_feasible=path_feasible,
                soc_feasible=soc_feasible,
                detour_feasible=detour_feasible,
                empty_mask=bool(trace.empty_mask),
            )
        )
    if len(records) != len(traces):
        raise RuntimeError("not every accepted request produced an evaluation record")
    return records


def _executed_soc_is_feasible(
    *, request: Any, plan: Any, completed: Sequence[Any], generator: Any
) -> bool:
    epsilon = float(getattr(generator, "SOC_EPSILON", 1e-9))
    validate_plan = getattr(generator, "validate_plan", None)
    if callable(validate_plan):
        try:
            validate_plan(request, plan)
        except (RuntimeError, TypeError, ValueError):
            return False

    spec = request.vehicle_spec
    soc_min = float(spec.soc_min)
    battery_capacity = float(spec.battery_capacity)
    rho_kwh_per_km = float(spec.rho_kwh_per_km)
    if (
        not math.isfinite(soc_min)
        or not math.isfinite(battery_capacity)
        or battery_capacity <= 0.0
        or not math.isfinite(rho_kwh_per_km)
        or rho_kwh_per_km < 0.0
    ):
        return False

    expected_targets = (float(plan.lambda1),) + (
        () if plan.s2 is None else (float(request.target_soc),)
    )
    if len(completed) != len(expected_targets):
        return False
    for item, expected_target in zip(completed, expected_targets):
        if item.start_soc is None or item.end_soc is None or item.target_soc is None:
            return False
        start_soc = float(item.start_soc)
        end_soc = float(item.end_soc)
        recorded_target = float(item.target_soc)
        if not all(
            math.isfinite(value)
            for value in (start_soc, end_soc, recorded_target, expected_target)
        ):
            return False
        if (
            start_soc < soc_min - epsilon
            or recorded_target < start_soc - epsilon
            or abs(recorded_target - expected_target) > epsilon
            or end_soc < expected_target - epsilon
        ):
            return False

    route_legs = tuple(plan.route.legs)
    if len(route_legs) != len(expected_targets) + 1:
        return False
    departure_soc = float(spec.initial_soc)
    for leg_index, leg in enumerate(route_legs):
        arrival_soc = departure_soc - (
            rho_kwh_per_km * (float(leg.distance_m) / 1_000.0) / battery_capacity
        )
        if not math.isfinite(arrival_soc) or arrival_soc < soc_min - epsilon:
            return False
        if leg_index < len(expected_targets):
            departure_soc = expected_targets[leg_index]
    return True


def _queue_length_at_arrival(target: Any, history: Sequence[Any]) -> int:
    arrival = float(target.arrival_time)
    return sum(
        1
        for item in history
        if int(item.station_id) == int(target.station_id)
        and int(item.vehicle_id) != int(target.vehicle_id)
        and float(item.arrival_time) <= arrival + 1e-9
        and float(item.start_time) > arrival + 1e-9
    )


def aggregate_hierarchical_evaluation(
    records: Iterable[HierarchicalEvaluationRecord],
    *,
    bootstrap_seed: int = 7,
    bootstrap_resamples: int = 10_000,
) -> list[HierarchicalEvaluationAggregate]:
    items = list(records)
    _validate_records(items)
    grouped: list[tuple[str, str, list[_MetricRow]]] = [
        ("overall", "all", [_MetricRow(item) for item in items])
    ]
    dimensions = (
        ("day", lambda item: str(item.day)),
        ("load_class", lambda item: str(item.load_class)),
        ("hour", lambda item: str(int(item.hour))),
        ("plan_type", lambda item: str(item.plan_type)),
    )
    for dimension, key_function in dimensions:
        buckets: dict[str, list[_MetricRow]] = {}
        for item in items:
            buckets.setdefault(key_function(item), []).append(_MetricRow(item))
        keys = sorted(
            buckets,
            key=(
                (lambda value: int(value))
                if dimension == "hour"
                else (lambda value: value)
            ),
        )
        grouped.extend((dimension, key, buckets[key]) for key in keys)
    station_buckets: dict[str, list[_MetricRow]] = {}
    for item in items:
        for visit in item.visits:
            station_buckets.setdefault(str(int(visit.station_id)), []).append(
                _MetricRow(item, visit)
            )
    grouped.extend(
        ("station", station_id, station_buckets[station_id])
        for station_id in sorted(station_buckets, key=int)
    )
    return [
        _summarize_group(
            dimension,
            value,
            group_items,
            bootstrap_seed=int(bootstrap_seed),
            bootstrap_resamples=int(bootstrap_resamples),
        )
        for dimension, value, group_items in grouped
    ]


def write_hierarchical_evaluation_csv(
    summaries: Iterable[HierarchicalEvaluationAggregate],
    output_csv: str | Path,
    *,
    metadata: HierarchicalEvaluationMetadata,
) -> Path:
    items = list(summaries)
    if not items:
        raise ValueError("summaries must contain at least one aggregate")
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_values = {
        "checkpoint": metadata.checkpoint,
        "split": metadata.split,
        "seed": int(metadata.seed),
        "detour_limit": float(metadata.detour_limit),
        "policy": metadata.policy,
        "bootstrap_seed": int(metadata.bootstrap_seed),
        "bootstrap_resamples": int(metadata.bootstrap_resamples),
    }
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=EVALUATION_CSV_FIELDS)
        writer.writeheader()
        for summary in items:
            writer.writerow({**metadata_values, **asdict(summary)})
    return path


def write_hierarchical_evaluation_json(
    summaries: Iterable[HierarchicalEvaluationAggregate],
    output_json: str | Path,
    *,
    metadata: HierarchicalEvaluationMetadata,
) -> Path:
    items = list(summaries)
    if not items:
        raise ValueError("summaries must contain at least one aggregate")
    payload = {
        "checkpoint": metadata.checkpoint,
        "split": metadata.split,
        "seed": int(metadata.seed),
        "detour_limit": float(metadata.detour_limit),
        "policy": metadata.policy,
        "fallback_count": int(metadata.fallback_count),
        "fallback_rate": float(metadata.fallback_rate),
        "request_count": int(metadata.request_count),
        "bootstrap_seed": int(metadata.bootstrap_seed),
        "bootstrap_resamples": int(metadata.bootstrap_resamples),
        "groups": [asdict(item) for item in items],
    }
    path = Path(output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return path


def write_hierarchical_request_csv(
    records: Iterable[HierarchicalEvaluationRecord],
    output_csv: str | Path,
    *,
    metadata: HierarchicalEvaluationMetadata,
) -> Path:
    items = list(records)
    _validate_records(items)
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=REQUEST_CSV_FIELDS)
        writer.writeheader()
        for item in items:
            writer.writerow(_request_output_row(item, visits_as_json=True))
    return path


def write_hierarchical_request_json(
    records: Iterable[HierarchicalEvaluationRecord],
    output_json: str | Path,
    *,
    metadata: HierarchicalEvaluationMetadata,
) -> Path:
    items = list(records)
    _validate_records(items)
    payload = {
        "checkpoint": metadata.checkpoint,
        "split": metadata.split,
        "seed": int(metadata.seed),
        "detour_limit": float(metadata.detour_limit),
        "policy": metadata.policy,
        "fallback_count": int(metadata.fallback_count),
        "fallback_rate": float(metadata.fallback_rate),
        "request_count": int(metadata.request_count),
        "bootstrap_seed": int(metadata.bootstrap_seed),
        "bootstrap_resamples": int(metadata.bootstrap_resamples),
        "requests": [
            _request_output_row(item, visits_as_json=False) for item in items
        ],
    }
    path = Path(output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return path


def _request_output_row(
    item: HierarchicalEvaluationRecord, *, visits_as_json: bool
) -> dict[str, Any]:
    visits = [asdict(visit) for visit in item.visits]
    return {
        "checkpoint": item.checkpoint,
        "split": item.split,
        "seed": int(item.seed),
        "detour_limit": float(item.detour_limit),
        "policy": item.policy,
        "vehicle_id": int(item.vehicle_id),
        "day": item.day,
        "load_class": item.load_class,
        "hour": int(item.hour),
        "s1_station_id": int(item.s1_station_id),
        "s2_station_id": (
            (None if not visits_as_json else "")
            if item.s2_station_id is None
            else int(item.s2_station_id)
        ),
        "plan_type": item.plan_type,
        "wait_minutes_total": float(item.wait_minutes),
        "grid_energy_kwh_total": float(item.grid_energy_kwh),
        "curtailment_kwh_total": float(item.curtailment_kwh),
        "detour_ratio": float(item.detour_ratio),
        "additional_stop": bool(item.additional_stop),
        "fallback_used": bool(item.fallback_used),
        "service_feasible": bool(item.service_feasible),
        "path_feasible": bool(item.path_feasible),
        "soc_feasible": bool(item.soc_feasible),
        "detour_feasible": bool(item.detour_feasible),
        "empty_mask": bool(item.empty_mask),
        ("visits_json" if visits_as_json else "visits"): (
            json.dumps(visits, separators=(",", ":")) if visits_as_json else visits
        ),
    }


def _validate_records(records: list[HierarchicalEvaluationRecord]) -> None:
    if not records:
        raise ValueError("records must contain at least one evaluation record")
    first = records[0]
    expected = (
        str(first.checkpoint),
        str(first.split),
        int(first.seed),
        float(first.detour_limit),
        str(first.policy),
    )
    for item in records[1:]:
        actual = (
            str(item.checkpoint),
            str(item.split),
            int(item.seed),
            float(item.detour_limit),
            str(item.policy),
        )
        if actual != expected:
            raise ValueError("records contain mixed evaluation run metadata")


def _summarize_group(
    dimension: str,
    value: str,
    records: list[_MetricRow],
    *,
    bootstrap_seed: int,
    bootstrap_resamples: int,
) -> HierarchicalEvaluationAggregate:
    waits = np.asarray([item.wait_minutes for item in records], dtype=np.float64)
    grid = np.asarray([item.grid_energy_kwh for item in records], dtype=np.float64)
    curtailment = np.asarray(
        [item.curtailment_kwh for item in records], dtype=np.float64
    )
    detours = np.asarray([item.record.detour_ratio for item in records], dtype=np.float64)
    count = len(records)
    fallback_count = sum(bool(item.record.fallback_used) for item in records)
    empty_mask_count = sum(bool(item.record.empty_mask) for item in records)
    hard_feasible = [
        bool(item.record.service_feasible)
        and bool(item.record.path_feasible)
        and bool(item.record.soc_feasible)
        and bool(item.record.detour_feasible)
        and not bool(item.record.empty_mask)
        for item in records
    ]
    wait_interval = _bootstrap_metric_rows(
        records,
        value=lambda row: row.wait_minutes,
        seed=bootstrap_seed,
        resamples=bootstrap_resamples,
    )
    grid_interval = _bootstrap_metric_rows(
        records,
        value=lambda row: row.grid_energy_kwh,
        seed=bootstrap_seed,
        resamples=bootstrap_resamples,
    )
    curtailment_interval = _bootstrap_metric_rows(
        records,
        value=lambda row: row.curtailment_kwh,
        seed=bootstrap_seed,
        resamples=bootstrap_resamples,
    )
    detour_interval = _bootstrap_metric_rows(
        records,
        value=lambda row: float(row.record.detour_ratio),
        seed=bootstrap_seed,
        resamples=bootstrap_resamples,
    )
    return HierarchicalEvaluationAggregate(
        group_dimension=dimension,
        group_value=value,
        request_count=count,
        wait_minutes_total=float(waits.sum()),
        wait_minutes_mean=float(waits.mean()),
        wait_minutes_p50=float(np.quantile(waits, 0.50)),
        wait_minutes_p95=float(np.quantile(waits, 0.95)),
        wait_minutes_p99=float(np.quantile(waits, 0.99)),
        wait_minutes_max=float(waits.max()),
        wait_over_15_rate=float(np.mean(waits > 15.0)),
        wait_over_30_rate=float(np.mean(waits > 30.0)),
        wait_over_60_rate=float(np.mean(waits > 60.0)),
        max_station_queue=max(int(item.station_queue_length) for item in records),
        grid_energy_kwh_total=float(grid.sum()),
        grid_energy_kwh_mean=float(grid.mean()),
        curtailment_kwh_total=float(curtailment.sum()),
        curtailment_kwh_mean=float(curtailment.mean()),
        detour_ratio_mean=float(detours.mean()),
        detour_ratio_p95=float(np.quantile(detours, 0.95)),
        detour_ratio_max=float(detours.max()),
        additional_stop_rate=float(
            np.mean([bool(item.record.additional_stop) for item in records])
        ),
        fallback_count=int(fallback_count),
        fallback_rate=float(fallback_count / count),
        service_feasible_rate=float(
            np.mean([bool(item.record.service_feasible) for item in records])
        ),
        path_feasible_rate=float(
            np.mean([bool(item.record.path_feasible) for item in records])
        ),
        soc_feasible_rate=float(
            np.mean([bool(item.record.soc_feasible) for item in records])
        ),
        detour_feasible_rate=float(
            np.mean([bool(item.record.detour_feasible) for item in records])
        ),
        hard_feasible_rate=float(np.mean(hard_feasible)),
        empty_mask_count=int(empty_mask_count),
        bootstrap_unit="day",
        bootstrap_days=int(wait_interval.days),
        bootstrap_seed=int(bootstrap_seed),
        bootstrap_resamples=int(bootstrap_resamples),
        wait_minutes_mean_ci_lower=float(wait_interval.lower),
        wait_minutes_mean_ci_upper=float(wait_interval.upper),
        grid_energy_kwh_mean_ci_lower=float(grid_interval.lower),
        grid_energy_kwh_mean_ci_upper=float(grid_interval.upper),
        curtailment_kwh_mean_ci_lower=float(curtailment_interval.lower),
        curtailment_kwh_mean_ci_upper=float(curtailment_interval.upper),
        detour_ratio_mean_ci_lower=float(detour_interval.lower),
        detour_ratio_mean_ci_upper=float(detour_interval.upper),
    )


def _bootstrap_metric_rows(
    rows: Sequence[_MetricRow],
    *,
    value: Callable[[_MetricRow], float],
    seed: int,
    resamples: int,
) -> DayBootstrapInterval:
    values_by_day: dict[str, list[float]] = {}
    for row in rows:
        values_by_day.setdefault(str(row.record.day), []).append(float(value(row)))
    day_summaries = {
        day: float(np.mean(values)) for day, values in values_by_day.items()
    }
    return bootstrap_day_confidence_interval(
        day_summaries,
        seed=int(seed),
        resamples=int(resamples),
    )


def _feasible_candidates(generator: Any, context: Any) -> list[_FeasibleCandidate]:
    station_ids = tuple(int(value) for value in generator.station_ids)
    context_station_ids = tuple(int(value) for value in context.station_ids)
    if context_station_ids != station_ids:
        raise ValueError("feasibility context station order does not match generator")
    none_index = len(station_ids)
    if int(generator.none_index) != none_index:
        raise ValueError("generator none_index does not match station count")

    candidates: list[_FeasibleCandidate] = []
    s1_context = generator.build_s1_context(context)
    for raw_s1_index in np.flatnonzero(s1_context.mask):
        s1_index = int(raw_s1_index)
        s2_context = generator.build_s2_context(context, s1_index)
        for raw_s2_index in np.flatnonzero(s2_context.mask):
            s2_index = int(raw_s2_index)
            route = s2_context.routes[s2_index]
            if route is None:
                raise RuntimeError(
                    "feasible S2 context contains a missing candidate route"
                )
            distance = float(route.distance_m)
            if not math.isfinite(distance) or distance < 0.0:
                raise ValueError("feasible candidate route distance must be finite")
            if s2_index == none_index:
                candidates.append(
                    _FeasibleCandidate(
                        action=HierarchicalAction(s1_index, s2_index, None),
                        route_distance_m=distance,
                        is_split=False,
                    )
                )
                continue
            lambda_context = generator.build_lambda_context(
                context, s1_index, s2_index
            )
            if lambda_context is None:
                raise RuntimeError(
                    "feasible split context is missing its lambda context"
                )
            for raw_lambda_index in np.flatnonzero(lambda_context.mask):
                candidates.append(
                    _FeasibleCandidate(
                        action=HierarchicalAction(
                            s1_index, s2_index, int(raw_lambda_index)
                        ),
                        route_distance_m=distance,
                        is_split=True,
                    )
                )
    if not candidates:
        raise RuntimeError("feasibility contexts contain no feasible charging action")
    return candidates


def _deterministic_candidate_key(candidate: _FeasibleCandidate) -> tuple[float, bool, int, int, int]:
    action = candidate.action
    return (
        float(candidate.route_distance_m),
        bool(candidate.is_split),
        int(action.s1_index),
        int(action.s2_index),
        -1 if action.lambda_index is None else int(action.lambda_index),
    )


__all__ = [
    "BaselineSelection",
    "DayBootstrapInterval",
    "EVALUATION_CSV_FIELDS",
    "EVALUATION_POLICY_LABELS",
    "EvaluationEpisodeSpec",
    "REQUEST_CSV_FIELDS",
    "HierarchicalEvaluationAggregate",
    "HierarchicalEvaluationMetadata",
    "HierarchicalPolicyEvaluationReport",
    "HierarchicalEvaluationRecord",
    "StationVisit",
    "aggregate_hierarchical_evaluation",
    "bootstrap_day_confidence_interval",
    "select_mandatory_service_shortest",
    "select_minimum_wait_single",
    "select_random_feasible",
    "run_hierarchical_evaluation",
    "write_hierarchical_evaluation_csv",
    "write_hierarchical_evaluation_json",
    "write_hierarchical_request_csv",
    "write_hierarchical_request_json",
]
