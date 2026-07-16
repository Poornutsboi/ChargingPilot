from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Callable, Protocol

import numpy as np

from chargingpilot.simulator.models import ChargingAssignment, StationSpec, VehicleSpec


class NetworkProtocol(Protocol):
    def path_time(
        self,
        u: int,
        v: int,
        t: float,
        route_nodes=None,
    ) -> float:
        ...

    def path_energy(
        self,
        u: int,
        v: int,
        t: float,
        vehicle_or_rho,
        route_nodes=None,
    ) -> float:
        ...


@dataclass(frozen=True)
class VehicleRequest:
    vehicle_id: int
    decision_time: float
    vehicle_spec: VehicleSpec
    target_soc: float


@dataclass(frozen=True)
class EpisodeData:
    station_specs: tuple[StationSpec, ...]
    vehicle_requests: tuple[VehicleRequest, ...]
    network: NetworkProtocol | None = None
    timestep_minutes: float = 1.0
    initial_state: dict | None = None


EpisodeFactory = Callable[[], EpisodeData]


@dataclass(frozen=True)
class RewardWeights:
    wait_time: float = 5.0
    charge_time: float = 0.5
    stop_count: float = 0.1
    grid_energy: float = 1.0
    renewable_curtailment: float = 5.0
    violation: float = 1000.0


@dataclass(frozen=True)
class RewardScales:
    wait_time_minutes: float = 60.0
    charge_time_minutes: float = 60.0
    stop_count: float = 2.0
    grid_energy_kwh: float = 50.0
    renewable_curtailment_kwh: float = 100.0
    violation: float = 1.0


@dataclass(frozen=True)
class HierarchicalRewardWeights:
    wait_time: float = 10.0
    grid_energy: float = 0.5
    renewable_curtailment: float = 0.5
    detour: float = 0.5
    additional_stop: float = 0.2

    def __post_init__(self) -> None:
        _validate_positive_fields(self)


@dataclass(frozen=True)
class HierarchicalRewardScales:
    wait_time_minutes: float = 60.0
    grid_energy_kwh: float = 50.0
    renewable_curtailment_kwh: float = 50.0
    detour_ratio: float = 0.60
    additional_stop: float = 1.0

    def __post_init__(self) -> None:
        _validate_positive_fields(self)


@dataclass(frozen=True)
class ObservationScales:
    max_battery_kwh: float = 150.0
    max_vehicle_power_kw: float = 500.0
    max_rho_kwh_per_km: float = 0.5
    max_wait_minutes: float = 240.0
    distance_ratio_clip: float = 4.0
    energy_ratio_clip: float = 2.0
    power_ratio_clip: float = 4.0

    def __post_init__(self) -> None:
        _validate_positive_fields(self)


@dataclass(frozen=True)
class HierarchicalSplitChargingEnvConfig:
    max_detour_ratio: float = 0.60
    soc_epsilon: float = 1e-9
    incoming_windows_minutes: tuple[float, float, float] = (15.0, 30.0, 60.0)
    reward_weights: HierarchicalRewardWeights = field(default_factory=HierarchicalRewardWeights)
    reward_scales: HierarchicalRewardScales = field(default_factory=HierarchicalRewardScales)
    observation_scales: ObservationScales = field(default_factory=ObservationScales)

    def __post_init__(self) -> None:
        _validate_positive_fields(self, exclude=("reward_weights", "reward_scales", "observation_scales"))
        if len(self.incoming_windows_minutes) != 3:
            raise ValueError("incoming_windows_minutes must contain exactly three values")
        if any(left >= right for left, right in zip(self.incoming_windows_minutes, self.incoming_windows_minutes[1:])):
            raise ValueError("incoming_windows_minutes must be strictly increasing")


@dataclass(frozen=True)
class SplitChargingEnvConfig:
    max_station_count: int = 0
    episode_horizon_minutes: float = 24.0 * 60.0
    max_travel_time_minutes: float = 240.0
    max_battery_kwh: float = 150.0
    max_power_kw: float = 500.0
    max_ess_kwh: float = 10000.0
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    reward_scales: RewardScales = field(default_factory=RewardScales)
    interval_reward_discount: float = 0.99
    interval_reward_time_unit_minutes: float = 1.0
    max_drain_steps: int = 20000


@dataclass(frozen=True)
class DecodedAction:
    action_id: int
    pair_index: int
    bin_index: int
    s1: int
    s2: int | None
    z1_target: float
    z2_target: float
    valid: bool


@dataclass
class PendingDecision:
    decision_id: int
    vehicle_id: int
    entry_time: float
    action_id: int
    s1: int
    s2: int | None
    z1_target: float
    z2_target: float
    stop_count: int
    assignments: list[ChargingAssignment] = field(default_factory=list)
    violation: float = 0.0


def as_float32(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _validate_positive_fields(instance: object, *, exclude: tuple[str, ...] = ()) -> None:
    for name, value in vars(instance).items():
        if name in exclude:
            continue
        if isinstance(value, tuple):
            for item in value:
                _validate_positive_number(item, f"{name} values")
        else:
            _validate_positive_number(value, name)


def _validate_positive_number(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
