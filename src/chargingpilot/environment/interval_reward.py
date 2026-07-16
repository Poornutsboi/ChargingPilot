from __future__ import annotations

from chargingpilot.environment.models import HierarchicalRewardScales, HierarchicalRewardWeights
from chargingpilot.simulator.models import IntervalMetrics


def interval_reward(
    delta: IntervalMetrics,
    detour_ratio: float,
    has_second_stop: bool,
    weights: HierarchicalRewardWeights,
    scales: HierarchicalRewardScales,
) -> float:
    """Return the approved five-term negative interval cost."""

    return -float(
        float(weights.wait_time)
        * float(delta.wait_vehicle_minutes)
        / float(scales.wait_time_minutes)
        + float(weights.grid_energy)
        * float(delta.grid_energy_kwh)
        / float(scales.grid_energy_kwh)
        + float(weights.renewable_curtailment)
        * float(delta.renewable_curtailed_kwh)
        / float(scales.renewable_curtailment_kwh)
        + float(weights.detour) * float(detour_ratio) / float(scales.detour_ratio)
        + float(weights.additional_stop)
        * float(bool(has_second_stop))
        / float(scales.additional_stop)
    )
