from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class VehicleStatus(Enum):
    QUEUEING = "queueing"
    CHARGING = "charging"
    COMPLETE = "complete"


@dataclass(frozen=True)
class StationSpec:
    station_id: int
    charge_capacity: int
    limits: dict | None = None
    p_plug_kw: float = 120.0
    p_max_kw: float = 120.0
    eta: float = 0.95
    power_trace: tuple[tuple[float, float], ...] | None = None
    power_sharing_disabled: bool = False
    p_grid_max_kw: float | None = None
    renewable_power_trace: tuple[tuple[float, float], ...] | None = None
    ess_capacity_kwh: float = 0.0
    ess_initial_kwh: float = 0.0
    ess_charge_efficiency: float = 0.95
    ess_discharge_efficiency: float = 0.95
    p_ess_charge_max_kw: float = 0.0
    p_ess_discharge_max_kw: float = 0.0
    ess_power_trace: tuple[tuple[float, float], ...] | None = None


@dataclass(frozen=True)
class VehicleSpec:
    battery_capacity: float
    initial_soc: float
    soc_min: float
    p_max_kw: float
    p_min_kw: float
    rho_kwh_per_km: float
    origin: int
    destination: int
    departure_time: float
    path_nodes: tuple[int, ...]
    path_edges: tuple[str, ...]
    candidate_stations: tuple[int, ...] = field(default_factory=tuple)
    demand_kwh: float | None = None


@dataclass(frozen=True)
class ChargingSocRequest:
    vehicle_id: int
    station_id: int
    arrival_time: float
    vehicle_spec: VehicleSpec
    arrival_soc: float
    target_soc: float

    def __post_init__(self) -> None:
        spec = self.vehicle_spec
        arrival_soc = float(self.arrival_soc)
        target_soc = float(self.target_soc)
        if arrival_soc < float(spec.soc_min):
            raise ValueError("arrival_soc must be >= vehicle soc_min.")
        if target_soc < arrival_soc:
            raise ValueError("target_soc must be >= arrival_soc.")


@dataclass(frozen=True)
class ChargingAssignment:
    vehicle_id: int
    station_id: int
    charger_id: int
    arrival_time: float
    start_time: float
    end_time: float
    wait_time: float
    status_at_arrival: VehicleStatus
    start_soc: float | None = None
    end_soc: float | None = None
    target_soc: float | None = None
    energy_delivered_kwh: float = 0.0
    renewable_used_kwh: float = 0.0
    grid_used_kwh: float = 0.0
    renewable_curtailed_kwh: float = 0.0


@dataclass(frozen=True)
class ChargingHistoryRecord:
    vehicle_id: int
    station_id: int
    charger_id: int
    arrival_time: float
    start_time: float
    end_time: float
    wait_time: float
    start_soc: float | None = None
    end_soc: float | None = None
    target_soc: float | None = None
    energy_delivered_kwh: float = 0.0
    renewable_used_kwh: float = 0.0
    grid_used_kwh: float = 0.0
    renewable_curtailed_kwh: float = 0.0


@dataclass(frozen=True)
class VehicleState:
    vehicle_id: int
    station_id: int
    arrival_time: float
    charge_duration: float
    start_time: float
    end_time: float
    wait_time: float
    status: VehicleStatus


@dataclass(frozen=True)
class StationState:
    station_id: int
    charge_capacity: int
    charger_status: list[float]
    available_info: list[bool]
    queue_waiting_time: list[float]
    queue_demand: list[float]
    active_vehicle_ids: list[int] = field(default_factory=list)
    active_soc: list[float] = field(default_factory=list)
    active_power_kw: list[float] = field(default_factory=list)
    power_available_kw: float = 0.0
    renewable_used_kw: float = 0.0
    ess_discharge_kw: float = 0.0
    ess_charge_kw: float = 0.0
    grid_used_kw: float = 0.0
    renewable_curtailed_kw: float = 0.0
    ess_energy_kwh: float = 0.0


@dataclass
class StationEnergyMetrics:
    renewable_used_kwh: float = 0.0
    ess_discharged_kwh: float = 0.0
    ess_charged_kwh: float = 0.0
    grid_used_kwh: float = 0.0
    renewable_curtailed_kwh: float = 0.0

    def copy(self) -> StationEnergyMetrics:
        return StationEnergyMetrics(
            renewable_used_kwh=float(self.renewable_used_kwh),
            ess_discharged_kwh=float(self.ess_discharged_kwh),
            ess_charged_kwh=float(self.ess_charged_kwh),
            grid_used_kwh=float(self.grid_used_kwh),
            renewable_curtailed_kwh=float(self.renewable_curtailed_kwh),
        )


@dataclass(frozen=True)
class IntervalMetrics:
    wait_vehicle_minutes: float = 0.0
    grid_energy_kwh: float = 0.0
    renewable_curtailed_kwh: float = 0.0


@dataclass
class SystemMetrics:
    ev_served: list[int]
    ev_queueing: list[int]
    queue_time: list[float]
    renewable_used_kwh: list[float] = field(default_factory=list)
    ess_discharged_kwh: list[float] = field(default_factory=list)
    ess_charged_kwh: list[float] = field(default_factory=list)
    grid_used_kwh: list[float] = field(default_factory=list)
    renewable_curtailed_kwh: list[float] = field(default_factory=list)

    def copy(self) -> SystemMetrics:
        return SystemMetrics(
            ev_served=list(self.ev_served),
            ev_queueing=list(self.ev_queueing),
            queue_time=list(self.queue_time),
            renewable_used_kwh=list(self.renewable_used_kwh),
            ess_discharged_kwh=list(self.ess_discharged_kwh),
            ess_charged_kwh=list(self.ess_charged_kwh),
            grid_used_kwh=list(self.grid_used_kwh),
            renewable_curtailed_kwh=list(self.renewable_curtailed_kwh),
        )


@dataclass(frozen=True)
class SystemState:
    clock: float
    stations: dict[int, StationState]
    metrics: SystemMetrics
    vehicles: dict[int, VehicleState] | None = None


@dataclass
class StationRuntimeSnapshot:
    station_id: int
    clock: float = 0.0
    waiting_queue: list[ChargingSocRequest] = field(default_factory=list)
    active_sessions: list[dict] = field(default_factory=list)
    ess_energy_kwh: float = 0.0
    renewable_used_kw: float = 0.0
    ess_discharge_kw: float = 0.0
    ess_charge_kw: float = 0.0
    grid_used_kw: float = 0.0
    renewable_curtailed_kw: float = 0.0
    power_available_kw: float = 0.0
    renewable_used_kwh: float = 0.0
    ess_discharged_kwh: float = 0.0
    ess_charged_kwh: float = 0.0
    grid_used_kwh: float = 0.0
    renewable_curtailed_kwh: float = 0.0
    wait_vehicle_minutes: float = 0.0


@dataclass
class VehicleRecord:
    assignment: ChargingAssignment

    def state_at(self, query_time: float) -> VehicleState | None:
        if float(query_time) < float(self.assignment.arrival_time):
            return None

        if float(query_time) < float(self.assignment.start_time):
            status = VehicleStatus.QUEUEING
        elif float(query_time) < float(self.assignment.end_time):
            status = VehicleStatus.CHARGING
        else:
            status = VehicleStatus.COMPLETE

        return VehicleState(
            vehicle_id=int(self.assignment.vehicle_id),
            station_id=int(self.assignment.station_id),
            arrival_time=float(self.assignment.arrival_time),
            charge_duration=float(self.assignment.end_time - self.assignment.start_time),
            start_time=float(self.assignment.start_time),
            end_time=float(self.assignment.end_time),
            wait_time=float(self.assignment.wait_time),
            status=status,
        )
