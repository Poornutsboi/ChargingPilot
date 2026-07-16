from __future__ import annotations

from dataclasses import dataclass

from chargingpilot.simulator.models import VehicleSpec


@dataclass(frozen=True)
class Commitment:
    vehicle_id: int
    source_station_id: int
    target_station_id: int
    departure_time: float
    departure_soc: float
    expected_arrival_time: float
    expected_arrival_soc: float
    target_soc: float
    vehicle_spec: VehicleSpec
    created_at: float

    @property
    def planned_second_charge_duration(self) -> float:
        return self.planned_second_charge_energy_kwh

    @property
    def planned_second_charge_energy_kwh(self) -> float:
        return max(
            0.0,
            (float(self.target_soc) - float(self.expected_arrival_soc))
            * float(self.vehicle_spec.battery_capacity),
        )


class CommitmentStore:
    def __init__(self, station_ids: list[int]) -> None:
        if not station_ids:
            raise ValueError("station_ids must not be empty.")
        self._station_ids = sorted(int(station_id) for station_id in station_ids)
        self._metric_size = 1 + max(self._station_ids)
        self._commitments: dict[int, Commitment] = {}

    def add(self, commitment: Commitment) -> None:
        self._commitments[int(commitment.vehicle_id)] = commitment

    def get(self, vehicle_id: int) -> Commitment | None:
        return self._commitments.get(int(vehicle_id))

    def pop(self, vehicle_id: int) -> Commitment:
        key = int(vehicle_id)
        if key not in self._commitments:
            raise KeyError(f"No active commitment for vehicle_id={key}.")
        return self._commitments.pop(key)

    def summary(self, now: float) -> dict:
        counts = [0 for _ in range(self._metric_size)]
        charge_demand = [0.0 for _ in range(self._metric_size)]
        earliest_eta = [-1.0 for _ in range(self._metric_size)]

        for commitment in self._commitments.values():
            station_id = int(commitment.target_station_id)
            counts[station_id] += 1
            charge_demand[station_id] += float(commitment.planned_second_charge_energy_kwh)
            eta = max(0.0, float(commitment.expected_arrival_time) - float(now))
            current_eta = earliest_eta[station_id]
            if current_eta < 0.0 or eta < current_eta:
                earliest_eta[station_id] = float(eta)

        return {
            "commitment_count": counts,
            "commitment_charge_demand": charge_demand,
            "earliest_expected_arrival_eta": earliest_eta,
        }

    def summary_extended(self, now: float, horizon: float = 15.0) -> dict:
        counts = [0 for _ in range(self._metric_size)]
        charge_demand_kwh = [0.0 for _ in range(self._metric_size)]
        arrival_soc_sum = [0.0 for _ in range(self._metric_size)]
        eta_min = [-1.0 for _ in range(self._metric_size)]
        eta_sum = [0.0 for _ in range(self._metric_size)]
        eta_count = [0 for _ in range(self._metric_size)]
        window_count = [0 for _ in range(self._metric_size)]

        now_value = float(now)
        horizon_value = max(0.0, float(horizon))

        for commitment in self._commitments.values():
            station_id = int(commitment.target_station_id)
            counts[station_id] += 1
            charge_demand_kwh[station_id] += float(commitment.planned_second_charge_energy_kwh)
            arrival_soc_sum[station_id] += float(commitment.expected_arrival_soc)
            eta = max(0.0, float(commitment.expected_arrival_time) - now_value)
            current_eta = eta_min[station_id]
            if current_eta < 0.0 or eta < current_eta:
                eta_min[station_id] = float(eta)
            eta_sum[station_id] += float(eta)
            eta_count[station_id] += 1
            if eta <= horizon_value:
                window_count[station_id] += 1

        eta_mean = [
            (eta_sum[index] / eta_count[index]) if eta_count[index] > 0 else -1.0
            for index in range(self._metric_size)
        ]
        arrival_soc_mean = [
            (arrival_soc_sum[index] / counts[index]) if counts[index] > 0 else 0.0
            for index in range(self._metric_size)
        ]

        return {
            "commit_count": counts,
            "commit_charge_demand_minutes": charge_demand_kwh,
            "commit_charge_demand_kwh": charge_demand_kwh,
            "commit_arrival_soc_mean": arrival_soc_mean,
            "commit_eta_min": eta_min,
            "commit_eta_mean": eta_mean,
            "commit_eta_window_count": window_count,
        }
