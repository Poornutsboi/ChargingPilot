from __future__ import annotations

from dataclasses import dataclass

from chargingpilot.simulator.battery import BATTERY_POWER_BREAKPOINTS, battery_power
from chargingpilot.simulator.models import (
    ChargingAssignment,
    ChargingSocRequest,
    IntervalMetrics,
    StationEnergyMetrics,
    StationRuntimeSnapshot,
    StationSpec,
    StationState,
    VehicleStatus,
)


EPS = 1e-9


@dataclass
class ActiveChargingSession:
    request: ChargingSocRequest
    charger_id: int
    start_time: float
    start_soc: float
    current_soc: float
    target_soc: float
    power_kw: float = 0.0
    energy_delivered_kwh: float = 0.0
    renewable_used_kwh: float = 0.0
    grid_used_kwh: float = 0.0
    renewable_curtailed_kwh: float = 0.0


@dataclass(frozen=True)
class StationEnergyLedger:
    renewable_used_kw: float = 0.0
    ess_discharge_kw: float = 0.0
    ess_charge_kw: float = 0.0
    grid_used_kw: float = 0.0
    renewable_curtailed_kw: float = 0.0
    power_available_kw: float = 0.0


def _trace_value_at(
    trace: tuple[tuple[float, float], ...] | None,
    t: float,
    default: float,
    *,
    clamp_nonnegative: bool = True,
) -> float:
    if not trace:
        value = float(default)
        return max(0.0, value) if clamp_nonnegative else value
    selected_value = float(trace[0][1])
    for start_time, value in trace:
        if float(t) < float(start_time):
            break
        selected_value = float(value)
    return max(0.0, selected_value) if clamp_nonnegative else selected_value


def _uses_energy_ledger(spec: StationSpec) -> bool:
    return (
        spec.p_grid_max_kw is not None
        or bool(spec.renewable_power_trace)
        or bool(spec.ess_power_trace)
        or float(spec.ess_capacity_kwh) > 0.0
        or float(spec.ess_initial_kwh) > 0.0
        or float(spec.p_ess_charge_max_kw) > 0.0
        or float(spec.p_ess_discharge_max_kw) > 0.0
    )


def _grid_limit(spec: StationSpec) -> float:
    if spec.p_grid_max_kw is None:
        return max(0.0, float(spec.p_max_kw))
    return max(0.0, float(spec.p_grid_max_kw))


def station_power_at(spec: StationSpec, t: float) -> float:
    if _uses_energy_ledger(spec):
        renewable_kw = _trace_value_at(spec.renewable_power_trace, float(t), 0.0)
        ess_discharge_kw = max(0.0, float(spec.p_ess_discharge_max_kw))
        if spec.ess_power_trace:
            ess_discharge_kw = min(
                ess_discharge_kw,
                max(
                    0.0,
                    _trace_value_at(
                        spec.ess_power_trace,
                        float(t),
                        0.0,
                        clamp_nonnegative=False,
                    ),
                ),
            )
        return max(
            0.0,
            _grid_limit(spec) + renewable_kw + ess_discharge_kw,
        )
    if not spec.power_trace:
        return max(0.0, float(spec.p_max_kw))
    return _trace_value_at(spec.power_trace, float(t), float(spec.p_max_kw))


def session_power(
    station_spec: StationSpec,
    vehicle_spec,
    soc: float,
    active_count: int,
    now: float,
) -> float:
    if int(active_count) <= 0:
        return 0.0
    if bool(station_spec.power_sharing_disabled):
        return max(
            0.0,
            min(
                float(battery_power(vehicle_spec, float(soc))),
                float(station_spec.p_plug_kw),
            ),
        )
    shared_power = station_power_at(station_spec, now) / float(active_count)
    return max(
        0.0,
        min(
            float(battery_power(vehicle_spec, float(soc))),
            float(station_spec.p_plug_kw),
            float(shared_power),
        ),
    )


class StationRuntime:
    def __init__(
        self,
        spec: StationSpec,
        timestep_minutes: float = 1.0,
        *,
        exact_internal_events: bool = False,
    ) -> None:
        if int(spec.charge_capacity) <= 0:
            raise ValueError("charge_capacity must be > 0.")
        if float(spec.p_plug_kw) <= 0.0:
            raise ValueError("p_plug_kw must be > 0.")
        if float(spec.eta) <= 0.0:
            raise ValueError("eta must be > 0.")
        if float(timestep_minutes) <= 0.0:
            raise ValueError("timestep_minutes must be > 0.")
        if spec.p_grid_max_kw is not None and float(spec.p_grid_max_kw) < 0.0:
            raise ValueError("p_grid_max_kw must be >= 0.")
        if float(spec.ess_capacity_kwh) < 0.0:
            raise ValueError("ess_capacity_kwh must be >= 0.")
        if float(spec.ess_initial_kwh) < 0.0:
            raise ValueError("ess_initial_kwh must be >= 0.")
        if float(spec.ess_initial_kwh) > float(spec.ess_capacity_kwh) + EPS:
            raise ValueError("ess_initial_kwh must be <= ess_capacity_kwh.")
        if float(spec.ess_charge_efficiency) <= 0.0:
            raise ValueError("ess_charge_efficiency must be > 0.")
        if float(spec.ess_discharge_efficiency) <= 0.0:
            raise ValueError("ess_discharge_efficiency must be > 0.")
        if float(spec.p_ess_charge_max_kw) < 0.0:
            raise ValueError("p_ess_charge_max_kw must be >= 0.")
        if float(spec.p_ess_discharge_max_kw) < 0.0:
            raise ValueError("p_ess_discharge_max_kw must be >= 0.")
        self.spec = spec
        self.timestep_minutes = float(timestep_minutes)
        self.exact_internal_events = bool(exact_internal_events)
        self._clock = 0.0
        self._waiting_queue: list[ChargingSocRequest] = []
        self._active_sessions: dict[int, ActiveChargingSession] = {}
        self._completed_assignments: dict[int, ChargingAssignment] = {}
        self._ess_energy_kwh = float(spec.ess_initial_kwh)
        self._current_energy_ledger = StationEnergyLedger(
            power_available_kw=station_power_at(spec, 0.0)
        )
        self._last_interval_energy_ledger = self._current_energy_ledger
        self._has_interval_energy_ledger = False
        self._energy_metrics = StationEnergyMetrics()
        self._wait_vehicle_minutes = 0.0

    @property
    def clock(self) -> float:
        return float(self._clock)

    def enqueue_soc_request(self, request: ChargingSocRequest) -> None:
        if int(request.station_id) != int(self.spec.station_id):
            raise ValueError("ChargingSocRequest station_id does not match this station.")
        self.advance_to(float(request.arrival_time))
        self._waiting_queue.append(request)
        self._start_waiting(float(request.arrival_time))
        self._redistribute(float(request.arrival_time))

    def submit_soc_request(self, request: ChargingSocRequest) -> ChargingAssignment:
        self.enqueue_soc_request(request)
        return self.advance_until_complete(int(request.vehicle_id))

    def drain_completed_assignments(self) -> list[ChargingAssignment]:
        assignments = sorted(
            self._completed_assignments.values(),
            key=lambda assignment: (float(assignment.end_time), int(assignment.vehicle_id)),
        )
        self._completed_assignments.clear()
        return assignments

    def advance_until_complete(self, vehicle_id: int) -> ChargingAssignment:
        vehicle_id = int(vehicle_id)
        while vehicle_id not in self._completed_assignments:
            if not self._active_sessions and not self._waiting_queue:
                raise RuntimeError("No active or waiting charging session is available.")
            self.step()
        return self._completed_assignments[vehicle_id]

    def advance_to(self, query_time: float) -> None:
        target_time = float(query_time)
        if target_time + EPS < self._clock:
            raise ValueError("query_time must be >= station clock.")
        if self.exact_internal_events:
            self._advance_interval(target_time)
            return
        while self._clock + self.timestep_minutes <= target_time + EPS:
            self.step()
        if target_time > self._clock + EPS:
            self._integrate_wait(target_time)
            self._clock = target_time
            self._start_waiting(target_time)
            self._redistribute(target_time)

    def step(self) -> list[ChargingAssignment]:
        if not self.exact_internal_events:
            return self._advance_fixed_timestep()
        return self._advance_interval(
            float(self._clock) + float(self.timestep_minutes)
        )

    def next_internal_event_time(self, limit_time: float) -> float:
        target = float(limit_time)
        if target + EPS < float(self._clock):
            raise ValueError("limit_time must be >= station clock.")
        if not self.exact_internal_events:
            return target
        return self._next_internal_event_time(target)

    def _advance_fixed_timestep(self) -> list[ChargingAssignment]:
        now = float(self._clock)
        next_time = now + float(self.timestep_minutes)
        self._start_waiting(now)
        self._redistribute(now)
        self._integrate_wait(next_time)
        self._advance_active_soc(to_time=next_time)
        self._clock = float(next_time)
        completed = self._complete_ready(next_time)
        self._start_waiting(next_time)
        self._redistribute(next_time)
        return completed

    def _advance_interval(self, target_time: float) -> list[ChargingAssignment]:
        completed: list[ChargingAssignment] = []
        while float(self._clock) + EPS < float(target_time):
            now = float(self._clock)
            self._start_waiting(now)
            self._redistribute(now)
            event_time = self._next_internal_event_time(float(target_time))
            self._integrate_wait(event_time)
            self._advance_active_soc(to_time=event_time)
            self._clock = float(event_time)
            completed.extend(self._complete_ready(event_time))
            self._start_waiting(event_time)
            self._redistribute(event_time)
        return completed

    def _next_internal_event_time(self, target_time: float) -> float:
        now = float(self._clock)
        candidates = [float(target_time)]
        for session in self._active_sessions.values():
            if float(session.power_kw) <= EPS:
                continue
            event_soc = float(session.target_soc)
            for breakpoint in BATTERY_POWER_BREAKPOINTS:
                if (
                    float(session.current_soc) + EPS < float(breakpoint)
                    < event_soc - EPS
                ):
                    event_soc = float(breakpoint)
                    break
            energy_kwh = (
                event_soc - float(session.current_soc)
            ) * float(session.request.vehicle_spec.battery_capacity)
            event_time = now + (
                energy_kwh
                * 60.0
                / (float(self.spec.eta) * float(session.power_kw))
            )
            if now + EPS < event_time < float(target_time) - EPS:
                candidates.append(float(event_time))
        for trace in (
            self.spec.power_trace,
            self.spec.renewable_power_trace,
            self.spec.ess_power_trace,
        ):
            if not trace:
                continue
            for start_time, _value in trace:
                value = float(start_time)
                if now + EPS < value < float(target_time) - EPS:
                    candidates.append(value)
                    break
        return min(candidates)

    def to_state(
        self,
        query_time: float,
        queue_waiting_time: list[float],
        queue_demand: list[float] | None = None,
    ) -> StationState:
        self.advance_to(float(query_time))
        active_items = [
            self._active_sessions[charger_id]
            for charger_id in sorted(self._active_sessions)
        ]
        charger_status = self._charger_status()
        own_waiting_time = [
            max(0.0, float(query_time) - float(request.arrival_time))
            for request in self._waiting_queue
        ]
        own_demand = [
            max(0.0, float(request.target_soc) - float(request.arrival_soc))
            for request in self._waiting_queue
        ]
        ledger = self._state_energy_ledger()
        return StationState(
            station_id=int(self.spec.station_id),
            charge_capacity=int(self.spec.charge_capacity),
            charger_status=charger_status,
            available_info=[bool(status <= EPS) for status in charger_status],
            queue_waiting_time=own_waiting_time + list(queue_waiting_time),
            queue_demand=own_demand + list(queue_demand or []),
            active_vehicle_ids=[int(session.request.vehicle_id) for session in active_items],
            active_soc=[float(session.current_soc) for session in active_items],
            active_power_kw=[float(session.power_kw) for session in active_items],
            power_available_kw=float(self._current_energy_ledger.power_available_kw),
            renewable_used_kw=float(ledger.renewable_used_kw),
            ess_discharge_kw=float(ledger.ess_discharge_kw),
            ess_charge_kw=float(ledger.ess_charge_kw),
            grid_used_kw=float(ledger.grid_used_kw),
            renewable_curtailed_kw=float(ledger.renewable_curtailed_kw),
            ess_energy_kwh=float(self._ess_energy_kwh),
        )

    def snapshot(self) -> StationRuntimeSnapshot:
        ledger = self._state_energy_ledger()
        return StationRuntimeSnapshot(
            station_id=int(self.spec.station_id),
            clock=float(self._clock),
            waiting_queue=list(self._waiting_queue),
            active_sessions=[
                {
                    "request": session.request,
                    "vehicle_id": int(session.request.vehicle_id),
                    "charger_id": int(session.charger_id),
                    "start_time": float(session.start_time),
                    "start_soc": float(session.start_soc),
                    "current_soc": float(session.current_soc),
                    "target_soc": float(session.target_soc),
                    "power_kw": float(session.power_kw),
                    "energy_delivered_kwh": float(session.energy_delivered_kwh),
                    "renewable_used_kwh": float(session.renewable_used_kwh),
                    "grid_used_kwh": float(session.grid_used_kwh),
                    "renewable_curtailed_kwh": float(session.renewable_curtailed_kwh),
                }
                for session in self._active_sessions.values()
            ],
            ess_energy_kwh=float(self._ess_energy_kwh),
            renewable_used_kw=float(ledger.renewable_used_kw),
            ess_discharge_kw=float(ledger.ess_discharge_kw),
            ess_charge_kw=float(ledger.ess_charge_kw),
            grid_used_kw=float(ledger.grid_used_kw),
            renewable_curtailed_kw=float(ledger.renewable_curtailed_kw),
            power_available_kw=float(self._current_energy_ledger.power_available_kw),
            renewable_used_kwh=float(self._energy_metrics.renewable_used_kwh),
            ess_discharged_kwh=float(self._energy_metrics.ess_discharged_kwh),
            ess_charged_kwh=float(self._energy_metrics.ess_charged_kwh),
            grid_used_kwh=float(self._energy_metrics.grid_used_kwh),
            renewable_curtailed_kwh=float(self._energy_metrics.renewable_curtailed_kwh),
            wait_vehicle_minutes=float(self._wait_vehicle_minutes),
        )

    def restore(self, snapshot: StationRuntimeSnapshot) -> None:
        self._clock = float(snapshot.clock)
        self._waiting_queue = list(snapshot.waiting_queue)
        self._ess_energy_kwh = min(
            float(self.spec.ess_capacity_kwh),
            max(0.0, float(snapshot.ess_energy_kwh)),
        )
        snapshot_ledger = StationEnergyLedger(
            renewable_used_kw=float(snapshot.renewable_used_kw),
            ess_discharge_kw=float(snapshot.ess_discharge_kw),
            ess_charge_kw=float(snapshot.ess_charge_kw),
            grid_used_kw=float(snapshot.grid_used_kw),
            renewable_curtailed_kw=float(snapshot.renewable_curtailed_kw),
            power_available_kw=float(snapshot.power_available_kw),
        )
        self._current_energy_ledger = snapshot_ledger
        self._last_interval_energy_ledger = snapshot_ledger
        self._has_interval_energy_ledger = True
        self._energy_metrics = StationEnergyMetrics(
            renewable_used_kwh=float(snapshot.renewable_used_kwh),
            ess_discharged_kwh=float(snapshot.ess_discharged_kwh),
            ess_charged_kwh=float(snapshot.ess_charged_kwh),
            grid_used_kwh=float(snapshot.grid_used_kwh),
            renewable_curtailed_kwh=float(snapshot.renewable_curtailed_kwh),
        )
        self._wait_vehicle_minutes = float(snapshot.wait_vehicle_minutes)
        self._active_sessions = {}
        for item in snapshot.active_sessions:
            request = item.get("request")
            if request is None:
                continue
            charger_id = int(item["charger_id"])
            self._active_sessions[charger_id] = ActiveChargingSession(
                request=request,
                charger_id=charger_id,
                start_time=float(item["start_time"]),
                start_soc=float(item["start_soc"]),
                current_soc=float(item["current_soc"]),
                target_soc=float(item["target_soc"]),
                power_kw=float(item.get("power_kw", 0.0)),
                energy_delivered_kwh=float(item.get("energy_delivered_kwh", 0.0)),
                renewable_used_kwh=float(item.get("renewable_used_kwh", 0.0)),
                grid_used_kwh=float(item.get("grid_used_kwh", 0.0)),
                renewable_curtailed_kwh=float(item.get("renewable_curtailed_kwh", 0.0)),
            )
        self._completed_assignments = {}
        self._redistribute(float(self._clock))

    def energy_metrics(self) -> StationEnergyMetrics:
        return self._energy_metrics.copy()

    def interval_metrics(self) -> IntervalMetrics:
        return IntervalMetrics(
            wait_vehicle_minutes=float(self._wait_vehicle_minutes),
            grid_energy_kwh=float(self._energy_metrics.grid_used_kwh),
            renewable_curtailed_kwh=float(
                self._energy_metrics.renewable_curtailed_kwh
            ),
        )

    def _integrate_wait(self, to_time: float) -> None:
        elapsed_minutes = max(0.0, float(to_time) - float(self._clock))
        self._wait_vehicle_minutes += float(len(self._waiting_queue)) * elapsed_minutes

    def _start_waiting(self, now: float) -> None:
        free_chargers = [
            charger_id
            for charger_id in range(int(self.spec.charge_capacity))
            if charger_id not in self._active_sessions
        ]
        while free_chargers and self._waiting_queue:
            request = self._waiting_queue.pop(0)
            charger_id = free_chargers.pop(0)
            self._active_sessions[charger_id] = ActiveChargingSession(
                request=request,
                charger_id=int(charger_id),
                start_time=float(now),
                start_soc=float(request.arrival_soc),
                current_soc=float(request.arrival_soc),
                target_soc=float(request.target_soc),
            )
            if float(request.arrival_soc) + EPS >= float(request.target_soc):
                self._complete_ready(float(now))
                free_chargers.append(int(charger_id))
                free_chargers.sort()

    def _advance_active_soc(self, to_time: float) -> None:
        elapsed_minutes = max(0.0, float(to_time) - float(self._clock))
        if elapsed_minutes <= EPS:
            return
        for session in self._active_sessions.values():
            if session.power_kw <= EPS:
                continue
            delivered_kwh = (
                float(self.spec.eta) * float(session.power_kw) * elapsed_minutes / 60.0
            )
            delta_soc = delivered_kwh / float(session.request.vehicle_spec.battery_capacity)
            new_soc = min(float(session.target_soc), float(session.current_soc) + delta_soc)
            actual_delta = new_soc - float(session.current_soc)
            session.energy_delivered_kwh += (
                actual_delta * float(session.request.vehicle_spec.battery_capacity)
            )
            session.current_soc = float(new_soc)
        self._apply_energy_ledger(elapsed_minutes)

    def _complete_ready(self, now: float) -> list[ChargingAssignment]:
        completed_chargers = [
            charger_id
            for charger_id, session in self._active_sessions.items()
            if float(session.current_soc) + EPS >= float(session.target_soc)
        ]
        assignments: list[ChargingAssignment] = []
        for charger_id in completed_chargers:
            session = self._active_sessions.pop(charger_id)
            request = session.request
            wait_time = float(session.start_time) - float(request.arrival_time)
            assignment = ChargingAssignment(
                vehicle_id=int(request.vehicle_id),
                station_id=int(request.station_id),
                charger_id=int(charger_id),
                arrival_time=float(request.arrival_time),
                start_time=float(session.start_time),
                end_time=float(now),
                wait_time=float(wait_time),
                status_at_arrival=(
                    VehicleStatus.QUEUEING if wait_time > EPS else VehicleStatus.CHARGING
                ),
                start_soc=float(session.start_soc),
                end_soc=float(session.current_soc),
                target_soc=float(session.target_soc),
                energy_delivered_kwh=float(session.energy_delivered_kwh),
                renewable_used_kwh=float(session.renewable_used_kwh),
                grid_used_kwh=float(session.grid_used_kwh),
                renewable_curtailed_kwh=float(session.renewable_curtailed_kwh),
            )
            self._completed_assignments[int(request.vehicle_id)] = assignment
            assignments.append(assignment)
        return sorted(
            assignments,
            key=lambda assignment: (float(assignment.end_time), int(assignment.vehicle_id)),
        )

    def _state_energy_ledger(self) -> StationEnergyLedger:
        if self._has_interval_energy_ledger:
            return self._last_interval_energy_ledger
        return self._current_energy_ledger

    def _apply_energy_ledger(self, elapsed_minutes: float) -> None:
        ledger = self._current_energy_ledger
        elapsed_hours = max(0.0, float(elapsed_minutes)) / 60.0
        if elapsed_hours <= EPS:
            return
        charged_kwh = (
            float(ledger.ess_charge_kw)
            * elapsed_hours
            * float(self.spec.ess_charge_efficiency)
        )
        discharged_kwh = (
            float(ledger.ess_discharge_kw)
            * elapsed_hours
            / float(self.spec.ess_discharge_efficiency)
        )
        self._ess_energy_kwh = min(
            float(self.spec.ess_capacity_kwh),
            max(0.0, float(self._ess_energy_kwh) + charged_kwh - discharged_kwh),
        )
        self._energy_metrics.renewable_used_kwh += (
            float(ledger.renewable_used_kw) * elapsed_hours
        )
        self._energy_metrics.ess_discharged_kwh += (
            float(ledger.ess_discharge_kw) * elapsed_hours
        )
        self._energy_metrics.ess_charged_kwh += (
            float(ledger.ess_charge_kw) * elapsed_hours
        )
        self._energy_metrics.grid_used_kwh += float(ledger.grid_used_kw) * elapsed_hours
        self._energy_metrics.renewable_curtailed_kwh += (
            float(ledger.renewable_curtailed_kw) * elapsed_hours
        )
        self._allocate_interval_energy_to_active_sessions(
            ledger=ledger,
            elapsed_hours=elapsed_hours,
        )
        self._last_interval_energy_ledger = ledger
        self._has_interval_energy_ledger = True

    def _allocate_interval_energy_to_active_sessions(
        self,
        *,
        ledger: StationEnergyLedger,
        elapsed_hours: float,
    ) -> None:
        total_power_kw = sum(float(session.power_kw) for session in self._active_sessions.values())
        if total_power_kw <= EPS:
            return
        renewable_kwh = float(ledger.renewable_used_kw) * float(elapsed_hours)
        grid_kwh = float(ledger.grid_used_kw) * float(elapsed_hours)
        curtailed_kwh = float(ledger.renewable_curtailed_kw) * float(elapsed_hours)
        for session in self._active_sessions.values():
            share = max(0.0, float(session.power_kw)) / total_power_kw
            session.renewable_used_kwh += renewable_kwh * share
            session.grid_used_kwh += grid_kwh * share
            session.renewable_curtailed_kwh += curtailed_kwh * share

    def _max_ess_discharge_power(self, elapsed_minutes: float) -> float:
        if float(self.spec.ess_capacity_kwh) <= EPS:
            return 0.0
        elapsed_hours = max(EPS, float(elapsed_minutes) / 60.0)
        energy_limited_kw = (
            float(self._ess_energy_kwh)
            * float(self.spec.ess_discharge_efficiency)
            / elapsed_hours
        )
        return max(
            0.0,
            min(float(self.spec.p_ess_discharge_max_kw), energy_limited_kw),
        )

    def _max_ess_charge_power(self, elapsed_minutes: float) -> float:
        if float(self.spec.ess_capacity_kwh) <= EPS:
            return 0.0
        elapsed_hours = max(EPS, float(elapsed_minutes) / 60.0)
        remaining_kwh = max(
            0.0,
            float(self.spec.ess_capacity_kwh) - float(self._ess_energy_kwh),
        )
        energy_limited_kw = (
            remaining_kwh
            / (elapsed_hours * float(self.spec.ess_charge_efficiency))
        )
        return max(0.0, min(float(self.spec.p_ess_charge_max_kw), energy_limited_kw))

    def _ess_trace_power_at(self, now: float) -> float | None:
        if not self.spec.ess_power_trace:
            return None
        return _trace_value_at(
            self.spec.ess_power_trace,
            float(now),
            0.0,
            clamp_nonnegative=False,
        )

    def _energy_capacity_for_vehicle_service(
        self,
        now: float,
        elapsed_minutes: float,
    ) -> tuple[float, float, float, float]:
        renewable_kw = _trace_value_at(self.spec.renewable_power_trace, float(now), 0.0)
        ess_discharge_limit_kw = self._max_ess_discharge_power(elapsed_minutes)
        ess_trace_power = self._ess_trace_power_at(now)
        if ess_trace_power is not None:
            ess_discharge_limit_kw = min(
                ess_discharge_limit_kw,
                max(0.0, float(ess_trace_power)),
            )
        grid_kw = _grid_limit(self.spec)
        total_kw = renewable_kw + ess_discharge_limit_kw + grid_kw
        return (
            max(0.0, total_kw),
            max(0.0, renewable_kw),
            max(0.0, ess_discharge_limit_kw),
            max(0.0, grid_kw),
        )

    def _allocate_energy_sources(
        self,
        *,
        vehicle_power_kw: float,
        renewable_kw: float,
        ess_discharge_limit_kw: float,
        grid_limit_kw: float,
        now: float,
        elapsed_minutes: float,
        power_available_kw: float,
    ) -> StationEnergyLedger:
        renewable_used_kw = min(float(renewable_kw), max(0.0, float(vehicle_power_kw)))
        remaining_vehicle_kw = max(0.0, float(vehicle_power_kw) - renewable_used_kw)
        ess_discharge_kw = min(float(ess_discharge_limit_kw), remaining_vehicle_kw)
        remaining_vehicle_kw = max(0.0, remaining_vehicle_kw - ess_discharge_kw)
        grid_used_kw = min(float(grid_limit_kw), remaining_vehicle_kw)

        surplus_renewable_kw = max(0.0, float(renewable_kw) - renewable_used_kw)
        ess_trace_power = self._ess_trace_power_at(now)
        ess_charge_limit_kw = self._max_ess_charge_power(elapsed_minutes)
        if ess_trace_power is not None:
            ess_charge_limit_kw = min(ess_charge_limit_kw, max(0.0, -ess_trace_power))
        ess_charge_kw = min(surplus_renewable_kw, ess_charge_limit_kw)
        renewable_curtailed_kw = max(0.0, surplus_renewable_kw - ess_charge_kw)

        return StationEnergyLedger(
            renewable_used_kw=float(renewable_used_kw),
            ess_discharge_kw=float(ess_discharge_kw),
            ess_charge_kw=float(ess_charge_kw),
            grid_used_kw=float(grid_used_kw),
            renewable_curtailed_kw=float(renewable_curtailed_kw),
            power_available_kw=float(power_available_kw),
        )

    def _redistribute(self, now: float) -> None:
        active_count = len(self._active_sessions)
        if _uses_energy_ledger(self.spec):
            self._redistribute_with_energy_ledger(float(now), active_count)
            return
        for session in self._active_sessions.values():
            session.power_kw = session_power(
                station_spec=self.spec,
                vehicle_spec=session.request.vehicle_spec,
                soc=float(session.current_soc),
                active_count=active_count,
                now=float(now),
            )
        self._current_energy_ledger = StationEnergyLedger(
            power_available_kw=station_power_at(self.spec, float(now))
        )

    def _redistribute_with_energy_ledger(self, now: float, active_count: int) -> None:
        (
            power_available_kw,
            renewable_kw,
            ess_discharge_limit_kw,
            grid_limit_kw,
        ) = self._energy_capacity_for_vehicle_service(
            now=float(now),
            elapsed_minutes=float(self.timestep_minutes),
        )

        if active_count <= 0:
            ledger = self._allocate_energy_sources(
                vehicle_power_kw=0.0,
                renewable_kw=renewable_kw,
                ess_discharge_limit_kw=ess_discharge_limit_kw,
                grid_limit_kw=grid_limit_kw,
                now=float(now),
                elapsed_minutes=float(self.timestep_minutes),
                power_available_kw=power_available_kw,
            )
            self._current_energy_ledger = ledger
            return

        if bool(self.spec.power_sharing_disabled):
            requested_powers = {
                charger_id: max(
                    0.0,
                    min(
                        float(battery_power(session.request.vehicle_spec, session.current_soc)),
                        float(self.spec.p_plug_kw),
                    ),
                )
                for charger_id, session in self._active_sessions.items()
            }
            total_requested = sum(requested_powers.values())
            scale = 1.0 if total_requested <= EPS else min(1.0, power_available_kw / total_requested)
            for charger_id, session in self._active_sessions.items():
                session.power_kw = float(requested_powers[charger_id] * scale)
        else:
            shared_power_kw = float(power_available_kw) / float(active_count)
            for session in self._active_sessions.values():
                session.power_kw = max(
                    0.0,
                    min(
                        float(battery_power(session.request.vehicle_spec, session.current_soc)),
                        float(self.spec.p_plug_kw),
                        shared_power_kw,
                    ),
                )

        total_vehicle_power_kw = sum(
            float(session.power_kw) for session in self._active_sessions.values()
        )
        self._current_energy_ledger = self._allocate_energy_sources(
            vehicle_power_kw=total_vehicle_power_kw,
            renewable_kw=renewable_kw,
            ess_discharge_limit_kw=ess_discharge_limit_kw,
            grid_limit_kw=grid_limit_kw,
            now=float(now),
            elapsed_minutes=float(self.timestep_minutes),
            power_available_kw=power_available_kw,
        )

    def _charger_status(self) -> list[float]:
        status: list[float] = []
        for charger_id in range(int(self.spec.charge_capacity)):
            session = self._active_sessions.get(charger_id)
            if session is None or session.power_kw <= EPS:
                status.append(0.0)
                continue
            remaining_soc = max(0.0, float(session.target_soc) - float(session.current_soc))
            continuous_remaining = (
                remaining_soc
                * float(session.request.vehicle_spec.battery_capacity)
                * 60.0
                / (float(self.spec.eta) * float(session.power_kw))
            )
            ticks = int(continuous_remaining // float(self.timestep_minutes))
            if continuous_remaining % float(self.timestep_minutes) > EPS:
                ticks += 1
            status.append(float(ticks * float(self.timestep_minutes)))
        return status
