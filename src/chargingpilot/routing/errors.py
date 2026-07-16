from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from chargingpilot.environment.models import VehicleRequest
    from .models import HierarchicalAction


class NoMandatoryServiceRouteError(RuntimeError):
    def __init__(self, origin: int, destination: int) -> None:
        self.origin = int(origin)
        self.destination = int(destination)
        super().__init__(
            f"no route from {self.origin} to {self.destination} visits a mandatory service station"
        )


class RouteCacheError(RuntimeError):
    def __init__(self, cache_path: Path, reason: str) -> None:
        self.cache_path = Path(cache_path)
        self.reason = str(reason)
        super().__init__(f"route cache {self.cache_path} is unusable: {self.reason}")


class NoFeasibleChargingPlanError(RuntimeError):
    def __init__(self, request: VehicleRequest, mask: np.ndarray) -> None:
        spec = request.vehicle_spec
        self.vehicle_id = int(request.vehicle_id)
        self.origin = int(spec.origin)
        self.destination = int(spec.destination)
        self.initial_soc = float(spec.initial_soc)
        self.target_soc = float(request.target_soc)
        self.mask = np.asarray(mask, dtype=np.bool_).copy()
        super().__init__(
            "no feasible charging plan for "
            f"vehicle={self.vehicle_id}, OD=({self.origin},{self.destination}), "
            f"SOC={self.initial_soc}, target={self.target_soc}"
        )


class InvalidHierarchicalActionError(RuntimeError):
    def __init__(
        self,
        request: VehicleRequest,
        action: HierarchicalAction,
        reason: str,
    ) -> None:
        spec = request.vehicle_spec
        self.vehicle_id = int(request.vehicle_id)
        self.origin = int(spec.origin)
        self.destination = int(spec.destination)
        self.initial_soc = float(spec.initial_soc)
        self.target_soc = float(request.target_soc)
        self.s1_index = int(action.s1_index)
        self.s2_index = int(action.s2_index)
        self.lambda_index = (
            None if action.lambda_index is None else int(action.lambda_index)
        )
        self.action = action
        self.reason = str(reason)
        super().__init__(
            "invalid hierarchical action for "
            f"vehicle={self.vehicle_id}, OD=({self.origin},{self.destination}), "
            f"SOC={self.initial_soc}, target={self.target_soc}, "
            f"action=({self.s1_index},{self.s2_index},{self.lambda_index}): "
            f"{self.reason}"
        )
