from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from chargingpilot.environment.models import ObservationScales, VehicleRequest
from chargingpilot.routing.models import RequestFeasibilityContext, RouteLeg, RouteResult
from chargingpilot.simulator.incoming import IncomingSummary
from chargingpilot.simulator.models import StationSpec


REQUEST_FEATURE_NAMES = (
    "start_soc",
    "target_soc",
    "minimum_soc",
    "battery_ratio",
    "vehicle_power_ratio",
    "rho_ratio",
    "time_of_day_sin",
    "time_of_day_cos",
    "base_distance_ratio",
    "direct_distance_ratio",
    "queued_chargers_ratio",
    "active_chargers_ratio",
    "grid_power_ratio",
    "renewable_power_ratio",
    "curtailed_power_ratio",
    "ess_energy_ratio",
)


STATION_FEATURE_NAMES = (
    "renewable_flag",
    "ess_flag",
    "capacity_ratio",
    "station_power_ratio",
    "grid_limit_ratio",
    "origin_station_distance_ratio",
    "station_destination_distance_ratio",
    "single_plan_distance_ratio",
    "origin_station_energy_ratio",
    "station_destination_energy_ratio",
    "reachable",
    "expected_arrival_time_sin",
    "expected_arrival_time_cos",
    "queue_ratio",
    "active_ratio",
    "available_ratio",
    "estimated_wait_ratio",
    "available_power_ratio",
    "renewable_used_ratio",
    "grid_used_ratio",
    "curtailed_ratio",
    "ess_energy_ratio",
    "renewable_arrival_ratio",
    "renewable_plus_30_ratio",
    "renewable_plus_60_ratio",
    "incoming_count_0_15_ratio",
    "incoming_count_15_30_ratio",
    "incoming_count_30_60_ratio",
    "incoming_kwh_0_15_ratio",
    "incoming_kwh_15_30_ratio",
    "incoming_kwh_30_60_ratio",
    "incoming_eta_min_ratio",
    "incoming_eta_mean_ratio",
)


@dataclass(frozen=True)
class StructuredObservation:
    origin_index: int
    destination_index: int
    request: np.ndarray
    stations: np.ndarray

    def __post_init__(self) -> None:
        if type(self.origin_index) is not int or self.origin_index < 0:
            raise ValueError("origin_index must be a nonnegative integer")
        if type(self.destination_index) is not int or self.destination_index < 0:
            raise ValueError("destination_index must be a nonnegative integer")
        object.__setattr__(
            self,
            "request",
            _immutable_float32("request", self.request, (len(REQUEST_FEATURE_NAMES),)),
        )
        object.__setattr__(
            self,
            "stations",
            _immutable_float32(
                "stations", self.stations, (72, len(STATION_FEATURE_NAMES))
            ),
        )


def build_structured_observation(
    *,
    request: VehicleRequest,
    now: float,
    simulator_state: Mapping[str, Any],
    station_specs: Sequence[StationSpec],
    oracle: Any,
    feasibility_context: RequestFeasibilityContext,
    incoming_summary: IncomingSummary,
    scales: ObservationScales,
) -> StructuredObservation:
    """Build the policy input from current state and accepted commitments only."""

    now_value = _finite_value(now, "now")
    specs = tuple(station_specs)
    station_ids = tuple(int(spec.station_id) for spec in specs)
    if len(station_ids) != 72:
        raise ValueError("structured observations require exactly 72 stations")
    if any(left >= right for left, right in zip(station_ids, station_ids[1:])):
        raise ValueError("station_specs must be strictly ascending by station_id")
    if station_ids != tuple(feasibility_context.station_ids):
        raise ValueError("station_specs must match feasibility-context station order")
    if station_ids != tuple(oracle.station_ids):
        raise ValueError("station_specs must match oracle station order")
    if station_ids != tuple(incoming_summary.station_ids):
        raise ValueError("incoming summary must match station order")

    node_ids = tuple(int(node_id) for node_id in oracle.node_ids)
    if any(left >= right for left, right in zip(node_ids, node_ids[1:])):
        raise ValueError("oracle graph-node mapping must be strictly ascending")
    expected_node_mapping = {node_id: index for index, node_id in enumerate(node_ids)}
    if dict(oracle.node_to_index) != expected_node_mapping:
        raise ValueError("oracle node_to_index must use ascending graph-node order")
    expected_station_mapping = {
        station_id: index for index, station_id in enumerate(station_ids)
    }
    if dict(oracle.station_to_index) != expected_station_mapping:
        raise ValueError("oracle station_to_index must use ascending station order")

    spec = request.vehicle_spec
    if int(spec.origin) not in expected_node_mapping:
        raise ValueError(f"origin {spec.origin} is not in the oracle graph")
    if int(spec.destination) not in expected_node_mapping:
        raise ValueError(f"destination {spec.destination} is not in the oracle graph")
    if int(feasibility_context.request.vehicle_id) != int(request.vehicle_id):
        raise ValueError("request does not match feasibility context")

    try:
        state_by_station = simulator_state["stations"]
    except (KeyError, TypeError) as exc:
        raise ValueError("simulator_state must contain stations") from exc
    normalized_state = {int(key): value for key, value in state_by_station.items()}
    if tuple(sorted(normalized_state)) != station_ids:
        raise ValueError("simulator state must contain exactly the configured stations")

    max_capacity = max((max(0, int(item.charge_capacity)) for item in specs), default=0)
    total_capacity = sum(max(0, int(item.charge_capacity)) for item in specs)
    total_station_power = sum(max(0.0, float(item.p_max_kw)) for item in specs)
    baseline_distance = max(0.0, float(feasibility_context.baseline.distance_m))
    max_network_distance = max(0.0, float(oracle.max_finite_distance_m))
    direct_distance = max(
        0.0,
        float(oracle.direct_route(int(spec.origin), int(spec.destination)).distance_m),
    )

    queued = 0
    active = 0
    system_grid_kw = 0.0
    system_renewable_kw = 0.0
    system_curtailed_kw = 0.0
    system_ess_kwh = 0.0
    system_ess_capacity_kwh = 0.0
    for station_spec in specs:
        station_state = normalized_state[int(station_spec.station_id)]
        queued += len(_state_value(station_state, "queue_demand", ()))
        active += len(_state_value(station_state, "active_vehicle_ids", ()))
        system_grid_kw += _nonnegative(_state_value(station_state, "grid_used_kw", 0.0))
        system_renewable_kw += _nonnegative(
            _state_value(station_state, "renewable_used_kw", 0.0)
        )
        system_curtailed_kw += _nonnegative(
            _state_value(station_state, "renewable_curtailed_kw", 0.0)
        )
        system_ess_kwh += _nonnegative(
            _state_value(station_state, "ess_energy_kwh", 0.0)
        )
        system_ess_capacity_kwh += max(
            0.0, float(station_spec.ess_capacity_kwh)
        )

    time_sin, time_cos = _time_encoding(now_value)
    request_features = np.asarray(
        [
            _finite_or_zero(spec.initial_soc),
            _finite_or_zero(request.target_soc),
            _finite_or_zero(spec.soc_min),
            _clipped_ratio(spec.battery_capacity, scales.max_battery_kwh),
            _clipped_ratio(spec.p_max_kw, scales.max_vehicle_power_kw),
            _clipped_ratio(spec.rho_kwh_per_km, scales.max_rho_kwh_per_km),
            time_sin,
            time_cos,
            _distance_ratio(baseline_distance, max_network_distance, scales),
            _distance_ratio(direct_distance, max_network_distance, scales),
            _load_ratio(queued, total_capacity, scales),
            _load_ratio(active, total_capacity, scales),
            _power_ratio(system_grid_kw, total_station_power, scales),
            _power_ratio(system_renewable_kw, total_station_power, scales),
            _power_ratio(system_curtailed_kw, total_station_power, scales),
            _load_ratio(system_ess_kwh, system_ess_capacity_kwh, scales),
        ],
        dtype=np.float32,
    )

    station_features = np.zeros((72, len(STATION_FEATURE_NAMES)), dtype=np.float32)
    battery_kwh = max(0.0, float(spec.battery_capacity))
    rho = max(0.0, float(spec.rho_kwh_per_km))
    for index, station_spec in enumerate(specs):
        station_id = int(station_spec.station_id)
        station_state = normalized_state[station_id]
        route = feasibility_context.single_routes.get(index)
        if route is None:
            try:
                route = oracle.route_via(
                    int(spec.origin), int(spec.destination), (station_id,)
                )
            except ValueError:
                route = None

        if route is None:
            origin_distance_m = 0.0
            destination_distance_m = 0.0
            route_distance_m = 0.0
            origin_energy_kwh = 0.0
            destination_energy_kwh = 0.0
            reachable = 0.0
            arrival_time = None
            arrival_sin = 0.0
            arrival_cos = 0.0
        else:
            origin_leg, destination_leg = _single_route_legs(
                route, int(spec.origin), station_id, int(spec.destination)
            )
            origin_distance_m = max(0.0, float(origin_leg.distance_m))
            destination_distance_m = max(0.0, float(destination_leg.distance_m))
            route_distance_m = max(0.0, float(route.distance_m))
            arrival_time = now_value + _exact_path_time(
                oracle=oracle,
                simulator_state=simulator_state,
                leg=origin_leg,
                departure_time=now_value,
            )
            arrival_sin, arrival_cos = _time_encoding(arrival_time)
            origin_energy_kwh = rho * origin_distance_m / 1000.0
            destination_energy_kwh = rho * destination_distance_m / 1000.0
            reachable = (
                1.0
                if _finite_or_zero(spec.initial_soc)
                - _safe_ratio(origin_energy_kwh, battery_kwh)
                >= _finite_or_zero(spec.soc_min) - 1e-9
                else 0.0
            )
        capacity = max(0, int(station_spec.charge_capacity))
        station_power = max(0.0, float(station_spec.p_max_kw))
        grid_limit = (
            station_power
            if station_spec.p_grid_max_kw is None
            else max(0.0, float(station_spec.p_grid_max_kw))
        )
        queue_count = len(_state_value(station_state, "queue_demand", ()))
        active_count = len(_state_value(station_state, "active_vehicle_ids", ()))
        available_count = sum(
            bool(value)
            for value in _state_value(station_state, "available_info", ())
        )
        estimated_wait = (
            0.0
            if arrival_time is None
            else _estimated_wait_minutes(
                station_state=station_state,
                capacity=capacity,
                now=now_value,
                arrival_time=arrival_time,
            )
        )
        ess_capacity = max(0.0, float(station_spec.ess_capacity_kwh))
        incoming_counts = incoming_summary.counts[index]
        incoming_kwh = incoming_summary.kwh[index]
        window_hours = (0.25, 0.25, 0.5)
        incoming_kwh_ratios = [
            _load_ratio(value, station_power * hours, scales)
            for value, hours in zip(incoming_kwh, window_hours)
        ]
        renewable_trace = station_spec.renewable_power_trace
        renewable_forecasts = (
            [0.0, 0.0, 0.0]
            if arrival_time is None
            else [
                _power_ratio(
                    _trace_value(renewable_trace, arrival_time + offset),
                    station_power,
                    scales,
                )
                for offset in (0.0, 30.0, 60.0)
            ]
        )

        values = [
            1.0 if renewable_trace is not None else 0.0,
            1.0 if ess_capacity > 0.0 else 0.0,
            _load_ratio(capacity, max_capacity, scales),
            _power_ratio(station_spec.p_plug_kw, station_power, scales),
            _power_ratio(grid_limit, station_power, scales),
            _distance_ratio(origin_distance_m, baseline_distance, scales),
            _distance_ratio(destination_distance_m, baseline_distance, scales),
            _distance_ratio(route_distance_m, baseline_distance, scales),
            _energy_ratio(origin_energy_kwh, battery_kwh, scales),
            _energy_ratio(destination_energy_kwh, battery_kwh, scales),
            reachable,
            arrival_sin,
            arrival_cos,
            _load_ratio(queue_count, capacity, scales),
            _load_ratio(active_count, capacity, scales),
            _load_ratio(available_count, capacity, scales),
            _load_ratio(estimated_wait, scales.max_wait_minutes, scales),
            _power_ratio(
                _state_value(station_state, "power_available_kw", 0.0),
                station_power,
                scales,
            ),
            _power_ratio(
                _state_value(station_state, "renewable_used_kw", 0.0),
                station_power,
                scales,
            ),
            _power_ratio(
                _state_value(station_state, "grid_used_kw", 0.0),
                station_power,
                scales,
            ),
            _power_ratio(
                _state_value(station_state, "renewable_curtailed_kw", 0.0),
                station_power,
                scales,
            ),
            _load_ratio(
                _state_value(station_state, "ess_energy_kwh", 0.0),
                ess_capacity,
                scales,
            ),
            *renewable_forecasts,
            *(_load_ratio(value, capacity, scales) for value in incoming_counts),
            *incoming_kwh_ratios,
            _load_ratio(incoming_summary.eta_min[index], 60.0, scales),
            _load_ratio(incoming_summary.eta_mean[index], 60.0, scales),
        ]
        station_features[index] = np.asarray(values, dtype=np.float32)

    if not np.isfinite(request_features).all() or not np.isfinite(station_features).all():
        raise ValueError("structured observation contains non-finite values")
    return StructuredObservation(
        origin_index=expected_node_mapping[int(spec.origin)],
        destination_index=expected_node_mapping[int(spec.destination)],
        request=request_features,
        stations=station_features,
    )


class StructuredObservationBuilder:
    """Named facade for callers that prefer a builder object."""

    @staticmethod
    def build(**kwargs: Any) -> StructuredObservation:
        return build_structured_observation(**kwargs)


def _single_route_legs(
    route: RouteResult,
    origin: int,
    station_id: int,
    destination: int,
) -> tuple[RouteLeg, RouteLeg]:
    if len(route.legs) != 2:
        raise ValueError(f"single-station route for {station_id} must have two legs")
    first, second = route.legs
    if (first.source, first.target) != (origin, station_id) or (
        second.source,
        second.target,
    ) != (station_id, destination):
        raise ValueError(f"single-station route for {station_id} has invalid legs")
    return first, second


def _exact_path_time(
    *,
    oracle: Any,
    simulator_state: Mapping[str, Any],
    leg: RouteLeg,
    departure_time: float,
) -> float:
    candidates = [oracle]
    for name in ("network", "_network"):
        value = getattr(oracle, name, None)
        if value is not None:
            candidates.append(value)
    state_network = simulator_state.get("network")
    if state_network is not None:
        candidates.append(state_network)
    for candidate in candidates:
        path_time = getattr(candidate, "path_time", None)
        if not callable(path_time):
            continue
        value = path_time(
            int(leg.source),
            int(leg.target),
            float(departure_time),
            route_nodes=tuple(leg.node_ids),
        )
        return max(0.0, _finite_value(value, "path time"))
    raise ValueError(
        "exact VDF path_time is unavailable; expose it on oracle or simulator_state['network']"
    )


def _estimated_wait_minutes(
    *, station_state: Any, capacity: int, now: float, arrival_time: float
) -> float:
    if capacity <= 0:
        return 0.0
    remaining = tuple(_state_value(station_state, "charger_status", ()))
    available = tuple(_state_value(station_state, "available_info", ()))
    heap: list[float] = []
    for charger_index in range(capacity):
        is_free = charger_index < len(available) and bool(available[charger_index])
        remaining_minutes = (
            _nonnegative(remaining[charger_index])
            if charger_index < len(remaining)
            else 0.0
        )
        heap.append(float(now) if is_free else float(now) + remaining_minutes)
    heapq.heapify(heap)
    for demand in _state_value(station_state, "queue_demand", ()):
        next_free = heapq.heappop(heap)
        duration = _nonnegative(demand)
        heapq.heappush(heap, max(float(now), next_free) + duration)
    return max(0.0, min(heap) - float(arrival_time))


def _trace_value(
    trace: tuple[tuple[float, float], ...] | None,
    query_time: float,
) -> float:
    if not trace:
        return 0.0
    value = float(trace[0][1])
    for start_time, candidate in trace:
        if float(query_time) < float(start_time):
            break
        value = float(candidate)
    return _nonnegative(value)


def _time_encoding(minutes: float) -> tuple[float, float]:
    angle = 2.0 * math.pi * (float(minutes) % 1440.0) / 1440.0
    return math.sin(angle), math.cos(angle)


def _distance_ratio(
    numerator: Any, denominator: Any, scales: ObservationScales
) -> float:
    value = _safe_ratio(numerator, denominator)
    return min(float(scales.distance_ratio_clip), max(0.0, value)) / float(
        scales.distance_ratio_clip
    )


def _energy_ratio(
    numerator: Any, denominator: Any, scales: ObservationScales
) -> float:
    value = _safe_ratio(numerator, denominator)
    return min(float(scales.energy_ratio_clip), max(0.0, value)) / float(
        scales.energy_ratio_clip
    )


def _power_ratio(
    numerator: Any, denominator: Any, scales: ObservationScales
) -> float:
    return min(
        float(scales.power_ratio_clip),
        max(0.0, _safe_ratio(numerator, denominator)),
    )


def _load_ratio(
    numerator: Any, denominator: Any, scales: ObservationScales
) -> float:
    return _power_ratio(numerator, denominator, scales)


def _clipped_ratio(numerator: Any, denominator: Any) -> float:
    return min(4.0, max(0.0, _safe_ratio(numerator, denominator)))


def _safe_ratio(numerator: Any, denominator: Any) -> float:
    bottom = _finite_or_zero(denominator)
    if bottom <= 0.0:
        return 0.0
    top = _finite_or_zero(numerator, positive_infinity=math.inf)
    return max(0.0, top / bottom)


def _state_value(state: Any, name: str, default: Any) -> Any:
    if isinstance(state, Mapping):
        return state.get(name, default)
    return getattr(state, name, default)


def _nonnegative(value: Any) -> float:
    return max(0.0, _finite_or_zero(value, positive_infinity=math.inf))


def _finite_or_zero(value: Any, *, positive_infinity: float = 0.0) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(normalized) or normalized == -math.inf:
        return 0.0
    if normalized == math.inf:
        return positive_infinity
    return normalized


def _finite_value(value: Any, name: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _immutable_float32(
    name: str,
    value: np.ndarray,
    shape: tuple[int, ...],
) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(np.float32):
        raise ValueError(f"{name} must be a float32 numpy array")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.frombuffer(value.tobytes(order="C"), dtype=np.float32).reshape(shape)
