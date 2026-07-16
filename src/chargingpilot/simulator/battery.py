from __future__ import annotations

from chargingpilot.simulator.models import VehicleSpec


BATTERY_POWER_SEGMENTS: tuple[tuple[float, float, float], ...] = (
    (0.00, 0.50, 1.00),
    (0.50, 0.70, 0.85),
    (0.70, 0.80, 0.60),
    (0.80, 0.90, 0.35),
    (0.90, 1.00, 0.15),
)

BATTERY_POWER_BREAKPOINTS: tuple[float, ...] = tuple(
    upper for _lower, upper, _factor in BATTERY_POWER_SEGMENTS[:-1]
)
SOC_BOUNDARY_TOL = 1e-9


def battery_power(spec: VehicleSpec, soc: float) -> float:
    soc_value = _snap_to_power_boundary(float(soc))
    if soc_value < float(spec.soc_min):
        raise ValueError(f"SOC {soc_value} is below vehicle soc_min {spec.soc_min}.")
    return float(spec.p_max_kw) * _power_factor_for_soc(soc_value)


def nominal_charge_duration_minutes(
    spec: VehicleSpec,
    target_soc: float | None = None,
) -> float:
    if target_soc is None:
        raise ValueError("target_soc is required.")
    target = float(target_soc)
    start = float(spec.initial_soc)
    if target <= start:
        return 0.0
    if start < float(spec.soc_min):
        raise ValueError(
            f"Charge interval start {start} is below vehicle soc_min {spec.soc_min}."
        )
    if float(spec.p_max_kw) <= 0.0 or float(spec.p_min_kw) <= 0.0:
        raise ValueError("Vehicle charging powers must be positive.")

    duration_hours = 0.0
    for lower, upper, factor in BATTERY_POWER_SEGMENTS:
        interval_start = max(start, float(lower))
        interval_end = min(target, float(upper))
        if interval_end <= interval_start:
            continue
        duration_hours += (
            float(spec.battery_capacity)
            * (interval_end - interval_start)
            / (float(spec.p_max_kw) * float(factor))
        )

    return float(duration_hours * 60.0)


def _power_factor_for_soc(soc: float) -> float:
    soc_value = float(soc)
    for index, (lower, upper, factor) in enumerate(BATTERY_POWER_SEGMENTS):
        if index == len(BATTERY_POWER_SEGMENTS) - 1:
            if float(lower) <= soc_value <= float(upper):
                return float(factor)
        elif float(lower) <= soc_value < float(upper):
            return float(factor)
    raise ValueError("SOC is outside the supported battery power curve [0.00, 1.00].")


def _snap_to_power_boundary(soc: float) -> float:
    soc_value = float(soc)
    for breakpoint in BATTERY_POWER_BREAKPOINTS:
        if abs(soc_value - float(breakpoint)) <= SOC_BOUNDARY_TOL:
            return float(breakpoint)
    return soc_value
