from __future__ import annotations

from chargingpilot.simulator.models import SystemMetrics


def build_empty_metrics(num_stations: int) -> SystemMetrics:
    return SystemMetrics(
        ev_served=[0 for _ in range(num_stations)],
        ev_queueing=[0 for _ in range(num_stations)],
        queue_time=[0.0 for _ in range(num_stations)],
        renewable_used_kwh=[0.0 for _ in range(num_stations)],
        ess_discharged_kwh=[0.0 for _ in range(num_stations)],
        ess_charged_kwh=[0.0 for _ in range(num_stations)],
        grid_used_kwh=[0.0 for _ in range(num_stations)],
        renewable_curtailed_kwh=[0.0 for _ in range(num_stations)],
    )
