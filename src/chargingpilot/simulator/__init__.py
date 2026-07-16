from chargingpilot.simulator.commitment import Commitment, CommitmentStore
from chargingpilot.simulator.history import ChargingHistoryLog
from chargingpilot.simulator.incoming import IncomingLoadTracker, IncomingRecord, IncomingSummary
from chargingpilot.simulator.models import (
    ChargingAssignment,
    ChargingHistoryRecord,
    ChargingSocRequest,
    StationSpec,
    StationState,
    SystemMetrics,
    SystemState,
    VehicleSpec,
    VehicleState,
    VehicleStatus,
)
from chargingpilot.simulator.planner import ChargingDecision, DecisionVehicle
from chargingpilot.simulator.simulator import SimulatorCore
from chargingpilot.simulator.station import StationRuntime, session_power, station_power_at


def __getattr__(name: str):
    if name in {"DemandForecaster", "SplitChargingOrchestrator"}:
        from chargingpilot.simulator.orchestrator import DemandForecaster, SplitChargingOrchestrator

        exports = {
            "DemandForecaster": DemandForecaster,
            "SplitChargingOrchestrator": SplitChargingOrchestrator,
        }
        return exports[name]
    raise AttributeError(f"module 'simulator' has no attribute {name!r}")

__all__ = [
    "ChargingAssignment",
    "ChargingHistoryLog",
    "ChargingHistoryRecord",
    "ChargingDecision",
    "ChargingSocRequest",
    "Commitment",
    "CommitmentStore",
    "DemandForecaster",
    "DecisionVehicle",
    "IncomingLoadTracker",
    "IncomingRecord",
    "IncomingSummary",
    "SimulatorCore",
    "StationRuntime",
    "SplitChargingOrchestrator",
    "StationSpec",
    "StationState",
    "SystemMetrics",
    "SystemState",
    "VehicleSpec",
    "VehicleState",
    "VehicleStatus",
    "session_power",
    "station_power_at",
]
