from __future__ import annotations

from dataclasses import dataclass

from chargingpilot.simulator.models import ChargingSocRequest, VehicleSpec


@dataclass(frozen=True)
class DecisionVehicle:
    vehicle_id: int
    station_id: int
    arrival_time: float
    total_charge_demand: float
    downstream_stations: tuple[int, ...] = ()
    vehicle_spec: VehicleSpec | None = None
    arrival_soc: float | None = None


@dataclass(frozen=True)
class ChargingDecision:
    second_station_id: int | None
    z1_target: float
    z2_target: float


@dataclass(frozen=True)
class SecondLegPlan:
    vehicle_id: int
    target_station_id: int
    z2_target: float


class SplitPlanner:
    def translate(
        self,
        current_ev: DecisionVehicle,
        decision: ChargingDecision,
    ) -> tuple[ChargingSocRequest, SecondLegPlan | None]:
        spec = current_ev.vehicle_spec
        if spec is None:
            raise ValueError("DecisionVehicle.vehicle_spec is required for SOC planning.")
        if current_ev.arrival_soc is None:
            raise ValueError("DecisionVehicle.arrival_soc is required for SOC planning.")

        arrival_soc = float(current_ev.arrival_soc)
        z1_target = float(decision.z1_target)
        z2_target = float(decision.z2_target)
        second_station_id = decision.second_station_id

        self._validate_target_soc(spec, arrival_soc, z1_target, "z1_target")

        first_request = ChargingSocRequest(
            vehicle_id=int(current_ev.vehicle_id),
            station_id=int(current_ev.station_id),
            arrival_time=float(current_ev.arrival_time),
            vehicle_spec=spec,
            arrival_soc=float(arrival_soc),
            target_soc=float(z1_target),
        )

        if second_station_id is None:
            return first_request, None

        if int(second_station_id) not in {int(station_id) for station_id in current_ev.downstream_stations}:
            raise ValueError("second_station_id must be one of the downstream stations.")
        if z2_target < float(spec.soc_min):
            raise ValueError("z2_target must be >= vehicle soc_min.")

        return (
            first_request,
            SecondLegPlan(
                vehicle_id=int(current_ev.vehicle_id),
                target_station_id=int(second_station_id),
                z2_target=float(z2_target),
            ),
        )

    def _validate_target_soc(
        self,
        spec: VehicleSpec,
        arrival_soc: float,
        target_soc: float,
        field_name: str,
    ) -> None:
        if arrival_soc < float(spec.soc_min):
            raise ValueError("arrival_soc must be >= vehicle soc_min.")
        if target_soc <= arrival_soc:
            raise ValueError(f"{field_name} must be greater than arrival_soc.")
