from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from chargingpilot.environment.models import VehicleRequest


@dataclass(frozen=True)
class RouteLeg:
    source: int
    target: int
    node_ids: tuple[int, ...]
    distance_m: float


@dataclass(frozen=True)
class RouteResult:
    required_station_ids: tuple[int, ...]
    legs: tuple[RouteLeg, ...]
    node_ids: tuple[int, ...]
    distance_m: float


@dataclass(frozen=True)
class ServiceBaseline:
    station_id: int
    distance_m: float
    node_ids: tuple[int, ...]


@dataclass(frozen=True)
class HierarchicalAction:
    s1_index: int
    s2_index: int
    lambda_index: int | None


@dataclass(frozen=True)
class ChargingPlan:
    vehicle_id: int
    s1: int
    s2: int | None
    lambda1: float
    baseline: ServiceBaseline
    route: RouteResult
    detour_ratio: float


@dataclass(frozen=True)
class RequestFeasibilityContext:
    request: VehicleRequest
    baseline: ServiceBaseline
    station_ids: tuple[int, ...]
    single_routes: dict[int, RouteResult]
    split_routes: dict[tuple[int, int], RouteResult]
    arrival_soc_s1: dict[int, float]

    def __post_init__(self) -> None:
        if not self.station_ids or any(
            left >= right
            for left, right in zip(self.station_ids, self.station_ids[1:])
        ):
            raise ValueError("station_ids must be strictly ascending and unique")


@dataclass(frozen=True)
class S1DecisionContext:
    request_context: RequestFeasibilityContext
    mask: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mask",
            _immutable_array(
                "S1DecisionContext.mask",
                self.mask,
                np.dtype(np.bool_),
                (len(self.request_context.station_ids),),
            ),
        )


@dataclass(frozen=True)
class S2DecisionContext:
    request_context: RequestFeasibilityContext
    s1_index: int
    mask: np.ndarray
    features: np.ndarray
    routes: tuple[RouteResult | None, ...]

    def __post_init__(self) -> None:
        size = len(self.request_context.station_ids) + 1
        object.__setattr__(
            self,
            "mask",
            _immutable_array(
                "S2DecisionContext.mask", self.mask, np.dtype(np.bool_), (size,)
            ),
        )
        object.__setattr__(
            self,
            "features",
            _immutable_array(
                "S2DecisionContext.features",
                self.features,
                np.dtype(np.float32),
                (size, 6),
            ),
        )
        if len(self.routes) != size:
            raise ValueError(f"S2DecisionContext.routes must have length {size}")


@dataclass(frozen=True)
class LambdaDecisionContext:
    request_context: RequestFeasibilityContext
    s1_index: int
    s2_index: int
    mask: np.ndarray
    features: np.ndarray
    bins: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mask",
            _immutable_array(
                "LambdaDecisionContext.mask", self.mask, np.dtype(np.bool_), (15,)
            ),
        )
        object.__setattr__(
            self,
            "features",
            _immutable_array(
                "LambdaDecisionContext.features",
                self.features,
                np.dtype(np.float32),
                (5,),
            ),
        )
        object.__setattr__(
            self,
            "bins",
            _immutable_array(
                "LambdaDecisionContext.bins",
                self.bins,
                np.dtype(np.float32),
                (15,),
            ),
        )


def _immutable_array(
    name: str,
    value: np.ndarray,
    dtype: np.dtype,
    shape: tuple[int, ...],
) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype != dtype or value.shape != shape:
        raise ValueError(f"{name} must be {dtype.name}{list(shape)}")
    return np.frombuffer(value.tobytes(order="C"), dtype=dtype).reshape(shape)
