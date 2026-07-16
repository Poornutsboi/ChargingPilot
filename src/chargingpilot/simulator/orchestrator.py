from __future__ import annotations

import ast
import csv
import math
from dataclasses import asdict, dataclass
from os import PathLike
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping

from data.network import RoadNetwork
from chargingpilot.simulator.commitment import Commitment, CommitmentStore
from chargingpilot.simulator.history import ChargingHistoryLog
from chargingpilot.simulator.models import ChargingAssignment, ChargingSocRequest, VehicleSpec
from chargingpilot.simulator.planner import ChargingDecision, DecisionVehicle, SplitPlanner
from chargingpilot.simulator.simulator import SimulatorCore

if TYPE_CHECKING:
    from envs.charging_env import Vehicle


@dataclass(frozen=True)
class PendingFirstLegPlan:
    vehicle_id: int
    source_station_id: int
    vehicle_spec: VehicleSpec
    target_station_id: int | None
    z2_target: float | None
    created_at: float


def _parse_station_sequence(value: Any, field_name: str) -> list[int]:
    if isinstance(value, str):
        parsed_value = ast.literal_eval(value)
    else:
        parsed_value = value

    if not isinstance(parsed_value, (list, tuple)):
        raise ValueError(f"Demand record {field_name} must be a list or tuple of station ids.")

    stations = [int(station_id) for station_id in parsed_value]
    if not stations:
        raise ValueError(f"Demand record {field_name} must contain at least one station.")
    return stations


def _station_sequence_from_record(record: Mapping[str, Any]) -> list[int]:
    if record.get("R") not in (None, ""):
        return _parse_station_sequence(record["R"], "R")
    if record.get("Route") not in (None, ""):
        return _parse_station_sequence(record["Route"], "Route")
    raise ValueError("Demand record is missing required station sequence field: R")


def _parse_int_sequence(value: Any, field_name: str) -> tuple[int, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        parsed_value = ast.literal_eval(value)
    else:
        parsed_value = value
    if not isinstance(parsed_value, (list, tuple)):
        raise ValueError(f"Demand record {field_name} must be a list or tuple.")
    return tuple(int(item) for item in parsed_value)


def _parse_str_sequence(value: Any, field_name: str) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        parsed_value = ast.literal_eval(value)
    else:
        parsed_value = value
    if not isinstance(parsed_value, (list, tuple)):
        raise ValueError(f"Demand record {field_name} must be a list or tuple.")
    return tuple(str(item) for item in parsed_value)


_REQUIRED_VEHICLE_SPEC_FIELDS = (
    "battery_capacity",
    "initial_soc",
    "rho_kwh_per_km",
)


def _target_soc_from_record(record: Mapping[str, Any]) -> float:
    if record.get("target_soc") not in (None, ""):
        return float(record["target_soc"])
    raise ValueError("Demand record is missing required VehicleSpec fields: target_soc")


def _resolved_path_from_record(record: Mapping[str, Any]) -> tuple[int, int, tuple[int, ...]]:
    explicit_path_nodes = _parse_int_sequence(record.get("path_nodes"), "path_nodes")
    if record.get("origin") not in (None, "") and record.get("destination") not in (None, ""):
        origin = int(record["origin"])
        destination = int(record["destination"])
        if explicit_path_nodes:
            return origin, destination, explicit_path_nodes
        from data.env_data import DEFAULT_ROAD_NETWORK

        return origin, destination, tuple(DEFAULT_ROAD_NETWORK._shortest_route(origin, destination))

    candidate_stations = tuple(_station_sequence_from_record(record))
    origin = int(record.get("origin") or candidate_stations[0])
    destination = int(record.get("destination") or candidate_stations[-1])
    return origin, destination, explicit_path_nodes or candidate_stations


def _vehicle_spec_from_record(record: Mapping[str, Any], candidate_stations: list[int]) -> VehicleSpec:
    missing = [
        field
        for field in _REQUIRED_VEHICLE_SPEC_FIELDS
        if record.get(field) in (None, "")
    ]
    if missing:
        raise ValueError(
            "Demand record is missing required VehicleSpec fields: "
            + ", ".join(missing)
        )

    origin, destination, path_nodes = _resolved_path_from_record(record)
    path_edges = _parse_str_sequence(record.get("path_edges"), "path_edges")
    if not path_edges and len(path_nodes) >= 2:
        from data.env_data import DEFAULT_ROAD_NETWORK

        try:
            path_edges = tuple(DEFAULT_ROAD_NETWORK.expand_route(path_nodes))
        except ValueError:
            path_edges = ()

    battery_capacity = float(record["battery_capacity"])
    initial_soc = float(record["initial_soc"])
    soc_min = float(record.get("soc_min") or 0.0)
    target_soc = _target_soc_from_record(record)
    expected_demand_kwh = max(0.0, battery_capacity * (target_soc - initial_soc))
    demand_kwh = (
        expected_demand_kwh
        if record.get("demand_kwh") in (None, "")
        else float(record["demand_kwh"])
    )
    if not math.isclose(float(demand_kwh), expected_demand_kwh, rel_tol=1e-5, abs_tol=1e-3):
        raise ValueError(
            "Demand record demand_kwh must equal battery_capacity * (target_soc - initial_soc)."
        )

    return VehicleSpec(
        battery_capacity=battery_capacity,
        initial_soc=initial_soc,
        soc_min=soc_min,
        p_max_kw=float(record.get("p_max_kw") or record.get("P_i_max") or 120.0),
        p_min_kw=float(record.get("p_min_kw") or 30.0),
        rho_kwh_per_km=float(record["rho_kwh_per_km"]),
        origin=origin,
        destination=destination,
        departure_time=float(record.get("departure_time") or record["Arrival_time"]),
        path_nodes=tuple(path_nodes),
        path_edges=tuple(path_edges),
        candidate_stations=tuple(candidate_stations),
        demand_kwh=float(demand_kwh),
    )


def demand_record_to_vehicle(record: Mapping[str, Any]) -> Vehicle:
    from envs.charging_env import Vehicle

    _origin, _destination, path_nodes = _resolved_path_from_record(record)
    candidate_stations = list(path_nodes)
    spec = _vehicle_spec_from_record(record, candidate_stations)
    return Vehicle(
        vid=int(record["Vehicle_ID"]),
        arrival_time=float(record["Arrival_time"]),
        spec=spec,
    )


def demand_records_to_vehicles(records: Iterable[Mapping[str, Any]]) -> list[Vehicle]:
    return [demand_record_to_vehicle(record) for record in records]


def load_demand_vehicles_from_csv(csv_path: str | PathLike[str]) -> list[Vehicle]:
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        return demand_records_to_vehicles(csv.DictReader(csv_file))


class DemandForecaster:
    def __init__(
        self,
        station_ids: list[int],
        station_capacities: dict[int, int] | None = None,
    ) -> None:
        self._station_ids = sorted(int(station_id) for station_id in station_ids)
        self._metric_size = 1 + max(self._station_ids)
        self._station_capacities = {
            int(station_id): int(capacity)
            for station_id, capacity in (station_capacities or {}).items()
        }

    def predict(
        self,
        method: str,
        now: float,
        history_log: ChargingHistoryLog,
        params: dict[str, float] | None = None,
    ) -> list[float]:
        if method == "exponential-decay":
            return self._predict_exponential_decay(
                now=now,
                history_log=history_log,
                params=params,
            )
        raise ValueError(f"Unsupported demand prediction method: {method}")

    def _predict_exponential_decay(
        self,
        now: float,
        history_log: ChargingHistoryLog,
        params : dict[str, float] | None = None,
    ) -> list[float]:
        config = params or {}
        horizon = float(config.get("horizon", 15.0))
        decay_tau = float(config.get("decay_tau", 15.0))
        weighted_counts = [0.0 for _ in range(self._metric_size)]

        for record in history_log.records():
            arrival_time = float(record.arrival_time)
            if arrival_time > float(now):
                continue
            elapsed = float(now) - arrival_time
            weighted_counts[int(record.station_id)] += math.exp(-elapsed / decay_tau)

        # weighted_counts[s] = 危 exp(-(now - t)/蟿); divide by 蟿 to get the per-time
        # arrival rate 位_s, multiply by the prediction window H to get the expected
        # number of arrivals in [now, now+H], then normalize by station capacity.
        rate_to_count = horizon / decay_tau
        result = [0.0 for _ in range(self._metric_size)]
        for station_id in range(self._metric_size):
            capacity = int(self._station_capacities.get(station_id, 1))
            if capacity <= 0:
                continue
            result[station_id] = weighted_counts[station_id] * rate_to_count / float(capacity)
        return result


class SplitChargingOrchestrator:
    def __init__(
        self,
        simulator: SimulatorCore,
        travel_time_estimator: Callable[[int, int], float] | None = None,
        network: RoadNetwork | None = None,
        planner: SplitPlanner | None = None,
        demand_prediction_method: str = "exponential-decay",
        demand_forecaster: DemandForecaster | None = None,
    ) -> None:
        self.simulator = simulator
        self.travel_time_estimator = travel_time_estimator or (lambda _a, _b: 0.0)
        self.network = network
        self.planner = planner or SplitPlanner()
        self.commitment_store = CommitmentStore(station_ids=self.simulator.station_ids)
        self._pending_first_leg_plans: dict[int, PendingFirstLegPlan] = {}
        self.demand_prediction_method = str(demand_prediction_method)
        self.demand_forecaster = demand_forecaster or DemandForecaster(
            station_ids=self.simulator.station_ids,
            station_capacities=self.simulator.station_capacities,
        )

    def build_observation(
        self,
        current_ev: DecisionVehicle,
        now: float,
        vehicle_info: bool = False,
    ) -> dict:
        return {
            "sim_state": self.simulator.get_state(query_time=now, vehicle_info=vehicle_info),
            "commitment_features": self.commitment_store.summary(now=now),
            "current_ev": asdict(current_ev),
            "future_demand": self.demand_forecaster.predict(
                method=self.demand_prediction_method,
                now=now,
                history_log=self.simulator.history_log,
            ),
            "travel_time_matrix": self._build_travel_time_matrix(),
        }

    def apply_decision(
        self,
        current_ev: DecisionVehicle,
        decision: ChargingDecision,
    ) -> dict:
        first_request, second_leg_plan = self.planner.translate(
            current_ev=current_ev,
            decision=decision,
        )
        self.simulator.enqueue_soc_arrival(first_request)

        spec = current_ev.vehicle_spec
        if spec is None:
            raise ValueError("vehicle_spec is required for SOC decisions.")
        pending_plan = PendingFirstLegPlan(
            vehicle_id=int(current_ev.vehicle_id),
            source_station_id=int(current_ev.station_id),
            vehicle_spec=spec,
            target_station_id=(
                None if second_leg_plan is None else int(second_leg_plan.target_station_id)
            ),
            z2_target=(
                None if second_leg_plan is None else float(second_leg_plan.z2_target)
            ),
            created_at=float(current_ev.arrival_time),
        )
        self._pending_first_leg_plans[int(current_ev.vehicle_id)] = pending_plan

        return {
            "first_request": first_request,
            "pending_plan": pending_plan,
            "commitment": None,
        }

    def handle_charging_completions(
        self,
        assignments: Iterable[ChargingAssignment],
    ) -> list[Commitment]:
        commitments: list[Commitment] = []
        for assignment in assignments:
            pending_plan = self._pending_first_leg_plans.pop(
                int(assignment.vehicle_id),
                None,
            )
            if pending_plan is None:
                continue
            if assignment.end_soc is None:
                raise ValueError("first-leg SOC assignment must include end_soc.")
            if pending_plan.target_station_id is None:
                self._validate_destination_reachable(
                    pending_plan=pending_plan,
                    departure_soc=float(assignment.end_soc),
                    departure_time=float(assignment.end_time),
                )
                continue
            commitment = self._schedule_second_leg_after_first_completion(
                pending_plan=pending_plan,
                first_assignment=assignment,
            )
            commitments.append(commitment)
        return commitments

    def _schedule_second_leg_after_first_completion(
        self,
        *,
        pending_plan: PendingFirstLegPlan,
        first_assignment: ChargingAssignment,
    ) -> Commitment:
        if self.network is None:
            raise ValueError("network is required for split SOC commitments.")
        if first_assignment.end_soc is None:
            raise ValueError("first assignment SOC is required for split commitments.")
        if pending_plan.target_station_id is None or pending_plan.z2_target is None:
            raise ValueError("split pending plan must include a target station and z2_target.")
        travel_time = float(
            self.network.path_time(
                int(pending_plan.source_station_id),
                int(pending_plan.target_station_id),
                float(first_assignment.end_time),
                route_nodes=pending_plan.vehicle_spec.path_nodes,
            )
        )
        expected_arrival_time = float(first_assignment.end_time) + travel_time
        expected_arrival_soc = self._arrival_soc_after_travel(
            spec=pending_plan.vehicle_spec,
            source_station_id=int(pending_plan.source_station_id),
            target_station_id=int(pending_plan.target_station_id),
            departure_soc=float(first_assignment.end_soc),
            departure_time=float(first_assignment.end_time),
        )
        commitment = Commitment(
            vehicle_id=int(pending_plan.vehicle_id),
            source_station_id=int(pending_plan.source_station_id),
            target_station_id=int(pending_plan.target_station_id),
            departure_time=float(first_assignment.end_time),
            departure_soc=float(first_assignment.end_soc),
            expected_arrival_time=float(expected_arrival_time),
            expected_arrival_soc=float(expected_arrival_soc),
            target_soc=float(pending_plan.z2_target),
            vehicle_spec=pending_plan.vehicle_spec,
            created_at=float(pending_plan.created_at),
        )
        self.commitment_store.add(commitment)
        self.simulator.schedule_soc_arrival(
            ChargingSocRequest(
                vehicle_id=int(commitment.vehicle_id),
                station_id=int(commitment.target_station_id),
                arrival_time=float(commitment.expected_arrival_time),
                vehicle_spec=commitment.vehicle_spec,
                arrival_soc=float(commitment.expected_arrival_soc),
                target_soc=float(commitment.target_soc),
            )
        )
        return commitment

    def _validate_destination_reachable(
        self,
        *,
        pending_plan: PendingFirstLegPlan,
        departure_soc: float,
        departure_time: float,
    ) -> None:
        if self.network is None:
            raise ValueError("network is required to validate destination reachability.")
        energy = float(
            self.network.path_energy(
                int(pending_plan.source_station_id),
                int(pending_plan.vehicle_spec.destination),
                float(departure_time),
                vehicle_or_rho=pending_plan.vehicle_spec,
                route_nodes=pending_plan.vehicle_spec.path_nodes,
            )
        )
        arrival_soc = float(departure_soc) - (
            energy / float(pending_plan.vehicle_spec.battery_capacity)
        )
        if arrival_soc + 1e-9 < float(pending_plan.vehicle_spec.soc_min):
            raise ValueError("no-split decision cannot reach destination after first charge.")

    def submit_scheduled_soc_arrival(
        self,
        request: ChargingSocRequest,
    ) -> None:
        self.commitment_store.pop(int(request.vehicle_id))
        self.simulator.enqueue_soc_arrival(request)

    def pending_first_leg_plans(self) -> tuple[PendingFirstLegPlan, ...]:
        return tuple(self._pending_first_leg_plans.values())

    def restore_pending_first_leg_plans(
        self,
        plans: Iterable[PendingFirstLegPlan],
    ) -> None:
        self._pending_first_leg_plans = {
            int(plan.vehicle_id): plan
            for plan in plans
        }

    def submit_second_leg_arrival(
        self,
        vehicle_id: int,
        actual_arrival_time: float,
    ) -> ChargingAssignment:
        commitment = self.commitment_store.pop(vehicle_id)
        arrival_soc = self._arrival_soc_after_travel(
            spec=commitment.vehicle_spec,
            source_station_id=int(commitment.source_station_id),
            target_station_id=int(commitment.target_station_id),
            departure_soc=float(commitment.departure_soc),
            departure_time=float(commitment.departure_time),
        )
        second_request = ChargingSocRequest(
            vehicle_id=int(commitment.vehicle_id),
            station_id=int(commitment.target_station_id),
            arrival_time=float(actual_arrival_time),
            vehicle_spec=commitment.vehicle_spec,
            arrival_soc=float(arrival_soc),
            target_soc=float(commitment.target_soc),
        )
        return self.simulator.submit_soc_arrival(second_request)

    def _arrival_soc_after_travel(
        self,
        *,
        spec: VehicleSpec,
        source_station_id: int,
        target_station_id: int,
        departure_soc: float,
        departure_time: float,
    ) -> float:
        if self.network is None:
            raise ValueError("network is required to calculate travel SOC.")
        energy = float(
            self.network.path_energy(
                int(source_station_id),
                int(target_station_id),
                float(departure_time),
                vehicle_or_rho=spec,
                route_nodes=spec.path_nodes,
            )
        )
        return float(departure_soc) - (energy / float(spec.battery_capacity))

    def _build_travel_time_matrix(self) -> list[list[float]]:
        station_ids = self.simulator.station_ids
        matrix_size = 1 + max(station_ids)
        matrix = [
            [0.0 for _ in range(matrix_size)]
            for _ in range(matrix_size)
        ]

        for from_station in station_ids:
            for to_station in station_ids:
                if from_station == to_station:
                    continue
                matrix[int(from_station)][int(to_station)] = float(
                    self.travel_time_estimator(int(from_station), int(to_station))
                )

        return matrix
