from __future__ import annotations

import heapq
import math
from dataclasses import asdict

from chargingpilot.simulator.history import ChargingHistoryLog
from chargingpilot.simulator.metrics import build_empty_metrics
from chargingpilot.simulator.models import (
    ChargingAssignment,
    ChargingHistoryRecord,
    ChargingSocRequest,
    IntervalMetrics,
    StationSpec,
    SystemMetrics,
    VehicleRecord,
    VehicleState,
    VehicleStatus,
)
from chargingpilot.simulator.station import StationRuntime


class ProjectionUnavailableError(RuntimeError):
    def __init__(self, station_id: int, vehicle_id: int, reason: str) -> None:
        self.station_id = int(station_id)
        self.vehicle_id = int(vehicle_id)
        self.reason = str(reason)
        super().__init__(
            "completion projection unavailable for "
            f"station={self.station_id}, vehicle={self.vehicle_id}: {self.reason}"
        )


class SimulatorCore:
    def __init__(
        self,
        station_specs: list[StationSpec],
        initial_state: dict | None = None,
        timestep_minutes: float = 1.0,
        *,
        exact_internal_events: bool = False,
    ) -> None:
        if not station_specs:
            raise ValueError("station_specs must not be empty.")
        if float(timestep_minutes) <= 0.0:
            raise ValueError("timestep_minutes must be > 0.")
        self.timestep_minutes = float(timestep_minutes)
        self._station_specs = {int(spec.station_id): spec for spec in station_specs}
        self._stations = {
            int(spec.station_id): StationRuntime(
                spec,
                timestep_minutes=self.timestep_minutes,
                exact_internal_events=bool(exact_internal_events),
            )
            for spec in station_specs
        }
        self._metric_size = 1 + max(int(spec.station_id) for spec in station_specs)
        self._metrics = build_empty_metrics(num_stations=self._metric_size)
        self._clock = 0.0
        self._future_soc_arrivals: list[tuple[float, int, ChargingSocRequest]] = []
        self._future_soc_arrival_sequence = 0
        self.history_log = ChargingHistoryLog()
        self._latest_record_by_vehicle: dict[int, VehicleRecord] = {}
        self._apply_initial_state(initial_state or {"stations": {}})

    @property
    def clock(self) -> float:
        return float(self._clock)

    @property
    def station_ids(self) -> list[int]:
        return sorted(int(station_id) for station_id in self._station_specs)

    @property
    def station_capacities(self) -> dict[int, int]:
        return {
            int(station_id): int(spec.charge_capacity)
            for station_id, spec in self._station_specs.items()
        }

    def submit_soc_arrival(self, request: ChargingSocRequest) -> ChargingAssignment:
        self.enqueue_soc_arrival(request)
        station = self._stations[int(request.station_id)]
        assignment = station.advance_until_complete(int(request.vehicle_id))
        completed = station.drain_completed_assignments()
        for completed_assignment in completed:
            self._record_soc_assignment(completed_assignment)
        self._clock = max(float(self._clock), float(assignment.end_time))
        return assignment

    def enqueue_soc_arrival(self, request: ChargingSocRequest) -> None:
        self._validate_soc_request(request)

        station = self._stations[int(request.station_id)]
        station.enqueue_soc_request(request)
        self._clock = max(float(self._clock), float(request.arrival_time))

    def advance_to(self, query_time: float) -> list[ChargingAssignment]:
        target_time = float(query_time)
        if target_time < float(self._clock):
            raise ValueError("query_time must be >= simulator clock.")

        completed: list[ChargingAssignment] = []
        for station in self._stations.values():
            station.advance_to(target_time)
            completed.extend(station.drain_completed_assignments())
        completed.sort(key=lambda assignment: (float(assignment.end_time), int(assignment.vehicle_id)))
        for assignment in completed:
            self._record_soc_assignment(assignment)
        self._clock = target_time
        return completed

    def next_internal_event_time(self, limit_time: float) -> float:
        target = float(limit_time)
        if target < float(self._clock):
            raise ValueError("limit_time must be >= simulator clock.")
        return min(
            station.next_internal_event_time(target)
            for station in self._stations.values()
        )

    def estimate_completion_time(self, request: ChargingSocRequest) -> float:
        """Project completion on an isolated station clone without mutating state."""
        self._validate_scheduled_soc_request(request)
        station_id = int(request.station_id)
        projection = StationRuntime(
            self._station_specs[station_id],
            timestep_minutes=self.timestep_minutes,
            exact_internal_events=True,
        )
        projection.restore(self._stations[station_id].snapshot())
        accepted = [
            (float(arrival_time), int(sequence), item)
            for arrival_time, sequence, item in self._future_soc_arrivals
            if int(item.station_id) == station_id
        ]
        candidate_sequence = max(
            (sequence for _time, sequence, _item in accepted), default=0
        ) + 1
        arrivals = sorted(
            [*accepted, (float(request.arrival_time), candidate_sequence, request)]
        )
        for arrival_time, _sequence, item in arrivals:
            projection.advance_to(arrival_time)
            for completed in projection.drain_completed_assignments():
                if int(completed.vehicle_id) == int(request.vehicle_id):
                    return float(completed.end_time)
            projection.enqueue_soc_request(item)
        while True:
            for completed in projection.drain_completed_assignments():
                if int(completed.vehicle_id) == int(request.vehicle_id):
                    return float(completed.end_time)
            event_time = projection.next_internal_event_time(math.inf)
            if not math.isfinite(event_time):
                raise ProjectionUnavailableError(
                    station_id,
                    int(request.vehicle_id),
                    "no future charging or power-change event",
                )
            if float(event_time) <= float(projection.clock) + 1e-9:
                raise ProjectionUnavailableError(
                    station_id,
                    int(request.vehicle_id),
                    f"no progress at time={projection.clock}",
                )
            projection.advance_to(event_time)

    def schedule_soc_arrival(self, request: ChargingSocRequest) -> None:
        self._validate_scheduled_soc_request(request)
        self._future_soc_arrival_sequence += 1
        heapq.heappush(
            self._future_soc_arrivals,
            (
                float(request.arrival_time),
                int(self._future_soc_arrival_sequence),
                request,
            ),
        )

    def next_scheduled_soc_arrival_time(self) -> float | None:
        if not self._future_soc_arrivals:
            return None
        return float(self._future_soc_arrivals[0][0])

    def pop_due_scheduled_soc_arrivals(self, query_time: float) -> list[ChargingSocRequest]:
        due: list[ChargingSocRequest] = []
        target_time = float(query_time)
        while (
            self._future_soc_arrivals
            and float(self._future_soc_arrivals[0][0]) <= target_time
        ):
            _time, _sequence, request = heapq.heappop(self._future_soc_arrivals)
            due.append(request)
        return due

    def scheduled_soc_arrivals(self) -> tuple[ChargingSocRequest, ...]:
        return tuple(
            request
            for _time, _sequence, request in sorted(self._future_soc_arrivals)
        )

    def get_state(
        self,
        query_time: float | None = None,
        vehicle_info: bool = False,
    ) -> dict:
        effective_time = float(self._clock if query_time is None else query_time)
        if effective_time < float(self._clock):
            raise ValueError("query_time must be >= the latest processed arrival_time.")

        queue_entries_by_station: dict[int, list[tuple[float, float, float]]] = {
            int(station_id): []
            for station_id in self._stations
        }
        current_queue_counts = [0 for _ in range(self._metric_size)]
        vehicle_states: dict[int, VehicleState] | None = {} if bool(vehicle_info) else None

        for vehicle_id, record in self._latest_record_by_vehicle.items():
            vehicle_state = record.state_at(effective_time)
            if vehicle_state is None:
                continue
            if vehicle_states is not None:
                vehicle_states[int(vehicle_id)] = vehicle_state
            if vehicle_state.status is VehicleStatus.QUEUEING:
                station_id = int(vehicle_state.station_id)
                queue_wait = float(
                    max(0.0, float(effective_time) - float(vehicle_state.arrival_time))
                )
                queue_entries_by_station[station_id].append(
                    (
                        float(vehicle_state.arrival_time),
                        queue_wait,
                        float(vehicle_state.charge_duration),
                    )
                )
                current_queue_counts[station_id] += 1

        metrics = self._metrics.copy()
        stations = {}
        for station_id, station in self._stations.items():
            ordered_entries = sorted(
                queue_entries_by_station[int(station_id)],
                key=lambda item: item[0],
            )
            stations[int(station_id)] = station.to_state(
                query_time=effective_time,
                queue_waiting_time=[entry[1] for entry in ordered_entries],
                queue_demand=[entry[2] for entry in ordered_entries],
            )

        metrics.ev_queueing = current_queue_counts
        self._fill_station_energy_metrics(metrics)

        state = {
            "clock": float(effective_time),
            "stations": {
                int(station_id): asdict(station_state)
                for station_id, station_state in stations.items()
            },
            "metrics": self._serialize_metrics(metrics),
        }
        if vehicle_states is not None:
            state["vehicles"] = {
                int(vehicle_id): self._serialize_vehicle_state(vehicle_state)
                for vehicle_id, vehicle_state in vehicle_states.items()
            }
        return state

    def get_metrics(self, query_time: float | None = None) -> SystemMetrics:
        effective_time = float(self._clock if query_time is None else query_time)
        if effective_time < float(self._clock):
            raise ValueError("query_time must be >= the latest processed arrival_time.")

        current_queue_counts = [0 for _ in range(self._metric_size)]
        for record in self._latest_record_by_vehicle.values():
            vehicle_state = record.state_at(effective_time)
            if vehicle_state is None:
                continue
            if vehicle_state.status is VehicleStatus.QUEUEING:
                current_queue_counts[int(vehicle_state.station_id)] += 1

        metrics = self._metrics.copy()
        metrics.ev_queueing = current_queue_counts
        self._fill_station_energy_metrics(metrics)
        return metrics

    def interval_metrics_snapshot(self) -> IntervalMetrics:
        snapshots = [station.interval_metrics() for station in self._stations.values()]
        return IntervalMetrics(
            wait_vehicle_minutes=sum(item.wait_vehicle_minutes for item in snapshots),
            grid_energy_kwh=sum(item.grid_energy_kwh for item in snapshots),
            renewable_curtailed_kwh=sum(
                item.renewable_curtailed_kwh for item in snapshots
            ),
        )

    def interval_metrics_delta(self, start: IntervalMetrics) -> IntervalMetrics:
        current = self.interval_metrics_snapshot()
        return IntervalMetrics(
            wait_vehicle_minutes=(
                current.wait_vehicle_minutes - float(start.wait_vehicle_minutes)
            ),
            grid_energy_kwh=current.grid_energy_kwh - float(start.grid_energy_kwh),
            renewable_curtailed_kwh=(
                current.renewable_curtailed_kwh
                - float(start.renewable_curtailed_kwh)
            ),
        )

    def _fill_station_energy_metrics(self, metrics: SystemMetrics) -> None:
        metrics.renewable_used_kwh = [0.0 for _ in range(self._metric_size)]
        metrics.ess_discharged_kwh = [0.0 for _ in range(self._metric_size)]
        metrics.ess_charged_kwh = [0.0 for _ in range(self._metric_size)]
        metrics.grid_used_kwh = [0.0 for _ in range(self._metric_size)]
        metrics.renewable_curtailed_kwh = [0.0 for _ in range(self._metric_size)]

        for station_id, station in self._stations.items():
            energy_metrics = station.energy_metrics()
            index = int(station_id)
            metrics.renewable_used_kwh[index] = float(
                energy_metrics.renewable_used_kwh
            )
            metrics.ess_discharged_kwh[index] = float(
                energy_metrics.ess_discharged_kwh
            )
            metrics.ess_charged_kwh[index] = float(energy_metrics.ess_charged_kwh)
            metrics.grid_used_kwh[index] = float(energy_metrics.grid_used_kwh)
            metrics.renewable_curtailed_kwh[index] = float(
                energy_metrics.renewable_curtailed_kwh
            )

    def _validate_soc_request(self, request: ChargingSocRequest) -> None:
        if int(request.station_id) not in self._stations:
            raise ValueError("station_id is not defined in this simulator.")
        if float(request.arrival_time) < float(self._clock):
            raise ValueError(
                "arrival_time must be non-decreasing because the simulator does not store future arrivals."
            )
        existing = self._latest_record_by_vehicle.get(int(request.vehicle_id))
        if existing is not None and float(request.arrival_time) < float(existing.assignment.end_time):
            raise ValueError("vehicle_id already has an unfinished charging reservation.")

    def _validate_scheduled_soc_request(self, request: ChargingSocRequest) -> None:
        if int(request.station_id) not in self._stations:
            raise ValueError("station_id is not defined in this simulator.")
        if float(request.arrival_time) < float(self._clock):
            raise ValueError("scheduled arrival_time must be >= simulator clock.")

    def _record_soc_assignment(self, assignment: ChargingAssignment) -> None:
        self.history_log.append(
            ChargingHistoryRecord(
                vehicle_id=int(assignment.vehicle_id),
                station_id=int(assignment.station_id),
                charger_id=int(assignment.charger_id),
                arrival_time=float(assignment.arrival_time),
                start_time=float(assignment.start_time),
                end_time=float(assignment.end_time),
                wait_time=float(assignment.wait_time),
                start_soc=assignment.start_soc,
                end_soc=assignment.end_soc,
                target_soc=assignment.target_soc,
                energy_delivered_kwh=float(assignment.energy_delivered_kwh),
                renewable_used_kwh=float(assignment.renewable_used_kwh),
                grid_used_kwh=float(assignment.grid_used_kwh),
                renewable_curtailed_kwh=float(assignment.renewable_curtailed_kwh),
            )
        )
        self._latest_record_by_vehicle[int(assignment.vehicle_id)] = VehicleRecord(
            assignment=assignment
        )
        self._metrics.ev_served[int(assignment.station_id)] += 1
        self._metrics.queue_time[int(assignment.station_id)] += float(assignment.wait_time)

    def _apply_initial_state(self, initial_state: dict) -> None:
        stations = dict(initial_state.get("stations", {}))
        if not stations:
            return
        for station_id in stations:
            if int(station_id) not in self._stations:
                raise ValueError("initial_state references an unknown station_id.")
        raise ValueError(
            "initial_state station bootstrap is no longer supported; SOC mode starts each "
            "station empty. Submit pre-existing sessions via enqueue_soc_arrival instead."
        )

    def _serialize_vehicle_state(self, vehicle_state: VehicleState) -> dict:
        payload = asdict(vehicle_state)
        payload["status"] = str(vehicle_state.status.value)
        return payload

    def _serialize_metrics(self, metrics: SystemMetrics) -> dict:
        return {
            "ev_served": list(metrics.ev_served),
            "ev_queueing": list(metrics.ev_queueing),
            "queue_time": list(metrics.queue_time),
            "renewable_used_kwh": list(metrics.renewable_used_kwh),
            "ess_discharged_kwh": list(metrics.ess_discharged_kwh),
            "ess_charged_kwh": list(metrics.ess_charged_kwh),
            "grid_used_kwh": list(metrics.grid_used_kwh),
            "renewable_curtailed_kwh": list(metrics.renewable_curtailed_kwh),
        }
